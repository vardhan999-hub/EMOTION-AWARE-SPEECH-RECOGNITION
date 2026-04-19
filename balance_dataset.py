# balance_dataset.py
"""
Balances the dataset manifest by downsampling over-represented classes.

Usage:
    python balance_dataset.py
"""
import pandas as pd

MANIFEST = "dataset_manifest.csv"


def balance():
    df = pd.read_csv(MANIFEST)
    print(f"[balance] Before:\n{df['emotion'].value_counts().to_string()}\n")

    counts    = df['emotion'].value_counts()
    target_n  = int(counts.median())   # target = median class size

    balanced = (
        df.groupby('emotion', group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), target_n), random_state=42))
          .sample(frac=1, random_state=42)
          .reset_index(drop=True)
    )

    balanced.to_csv(MANIFEST, index=False)
    print(f"[balance] After:\n{balanced['emotion'].value_counts().to_string()}")
    print(f"\n[balance] Total samples: {len(balanced)}")


if __name__ == "__main__":
    balance()
