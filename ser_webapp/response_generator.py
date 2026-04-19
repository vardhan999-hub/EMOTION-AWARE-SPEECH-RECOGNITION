# response_generator.py
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import DEVICE

MODEL_NAME = "microsoft/DialoGPT-small"

# ── Curated empathetic seed prompts per emotion ──────────────────
PROMPTS = {
    "happy": [
        "You sound really happy! That's wonderful — what's making your day so great?",
        "I can sense joy in your voice! Tell me more about it!",
        "Glad to hear you're in good spirits — anything special happened today?",
        "Your voice is full of positivity! Keep that energy going!",
        "Happiness suits you well! Care to share what made you smile?",
        "That joy is contagious! How do you usually celebrate good moods?",
        "That's great! What do you usually do when you feel this happy?",
        "I love hearing that happiness — what brought it on?",
        "You seem cheerful — what's been the best part of your day?",
        "That sounds awesome! It's nice to feel happy sometimes, right?",
    ],
    "sad": [
        "I'm sorry to hear you sound sad. Want to talk about it?",
        "I can hear a bit of sadness — what's been on your mind?",
        "It's okay to feel down sometimes. I'm here to listen.",
        "That sounds tough. Do you want to tell me what happened?",
        "I'm here if you need someone to talk to. What's making you feel this way?",
        "Feeling sad is human — would you like to share your thoughts?",
        "That sounds hard. How are you coping right now?",
        "It's alright to feel low sometimes — I've got your back.",
        "I can sense some sadness. Remember, better days are coming.",
        "You're not alone. Do you want to talk through what's bothering you?",
    ],
    "angry": [
        "You sound upset — what's bothering you?",
        "I can hear the frustration. Want to vent about it?",
        "Anger can be heavy — would you like to explain what happened?",
        "That sounds frustrating. Do you want to cool off or talk it out?",
        "It's okay to feel angry sometimes — what triggered it?",
        "I get that you're upset. How can I help calm things down?",
        "That must have annoyed you. Want to share more?",
        "Anger often hides hurt — do you feel misunderstood?",
        "I understand your frustration. What can we do about it?",
        "It's good to express feelings. Let's unpack that anger together.",
    ],
    "neutral": [
        "Thanks for sharing. Would you like to continue chatting?",
        "Okay, I'm listening — what's next?",
        "I see. Would you like to talk about something else?",
        "Alright, I'm here for whatever you want to discuss.",
        "Got it. Anything interesting you'd like to add?",
        "Hmm, sounds neutral — what are you thinking about?",
        "Okay. Should we dive deeper into that?",
        "Cool. What else is on your mind?",
        "Alright — anything you'd like to explore together?",
        "Understood. What topic would you like to shift to?",
    ],
    "fear": [
        "That sounds scary. Do you want to share what happened?",
        "Fear is a strong feeling — what's worrying you right now?",
        "You sound a bit anxious — want to talk about it?",
        "It's okay to be afraid sometimes. What's making you feel uneasy?",
        "I can sense fear — do you want to walk through it together?",
        "You're safe here. What's been bothering you lately?",
        "That sounds tense. Want to talk about your worries?",
        "I can hear anxiety in your tone — what's causing it?",
        "It's brave of you to share that fear. How can I support you?",
        "Sometimes fear teaches us something important — what do you think it's telling you?",
    ],
    "disgust": [
        "That sounds unpleasant — what made you feel this way?",
        "You seem disgusted — what happened?",
        "That reaction sounds intense. Want to explain why?",
        "I can hear disgust — was it something someone said or did?",
        "Sounds like you disliked that strongly — what triggered it?",
        "Disgust can be hard to process — want to talk about it?",
        "You seem uncomfortable. Care to share what caused it?",
        "That must have been unpleasant! What's your take on it?",
        "I hear disgust in your tone — want to discuss it more?",
        "Hmm, seems like that didn't sit well with you — why do you think?",
    ],
    "surprise": [
        "Wow, you sound surprised! What happened?",
        "That caught you off guard, didn't it?",
        "I can hear surprise — was it good or bad?",
        "That's unexpected! Tell me more!",
        "You seem shocked — what's the story?",
        "Whoa! Sounds like something surprising just happened!",
        "That's a twist! What did you think of it?",
        "You didn't see that coming, huh?",
        "Interesting! How did you react to that surprise?",
        "That's exciting! Want to share more about it?",
    ],
}


class EmotionResponder:
    """
    Generates empathetic responses based on detected emotion.
    Uses curated seed prompts + DialoGPT-small for natural continuation.
    Falls back to seed prompt if model unavailable.
    """

    def __init__(self, use_model: bool = False):
        self.use_model = use_model
        self._used: dict[str, list] = {e: [] for e in PROMPTS}

        if use_model:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
                self.model.eval()
                print(f"[response_generator] DialoGPT loaded on {DEVICE}")
            except Exception as e:
                print(f"[response_generator] Model load failed ({e}), using fallback.")
                self.use_model = False

    def _pick_seed(self, emotion: str) -> str:
        """Pick an unused seed prompt, resetting when all are used."""
        pool = PROMPTS.get(emotion.lower(), PROMPTS["neutral"])
        used = self._used.get(emotion.lower(), [])

        if len(used) >= len(pool):
            self._used[emotion.lower()] = []
            used = []

        available = [r for r in pool if r not in used]
        chosen    = random.choice(available)
        self._used[emotion.lower()].append(chosen)
        return chosen

    def generate(self, emotion: str) -> str:
        seed = self._pick_seed(emotion)

        if not self.use_model:
            return seed

        try:
            enc = self.tokenizer.encode(
                seed + self.tokenizer.eos_token, return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                out = self.model.generate(
                    enc,
                    max_new_tokens=40,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                )

            reply = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # Return only the generated continuation (after the seed)
            if seed in reply:
                reply = reply.replace(seed, "").strip()
            return reply if len(reply) > 10 else seed

        except Exception as e:
            print(f"[response_generator] Generation error: {e}")
            return seed
