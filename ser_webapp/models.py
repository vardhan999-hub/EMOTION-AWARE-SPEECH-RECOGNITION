# models.py
import torch
import torch.nn as nn


class TFAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.freq_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.time_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.freq_fc(x) * self.time_fc(x)


class CNN_TFA(nn.Module):
    def __init__(self, n_mels=64, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)),
        )
        self.tfa         = TFAttention(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc          = nn.Linear(64, out_dim)
        self.dropout     = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.tfa(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.dropout(self.fc(x))


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, H):
        score   = torch.tanh(self.w(H))
        e       = self.v(score).squeeze(-1)
        alpha   = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        return context, alpha


class BiLSTM_A(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.att     = AttentionLayer(hidden_dim)
        self.fc      = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x          = x.transpose(1, 2)
        H, _       = self.lstm(x)
        context, _ = self.att(H)
        return self.dropout(self.fc(context))


class FusionNet(nn.Module):
    def __init__(self, cnn_dim=256, lstm_dim=256, hidden=256, num_classes=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cnn_dim + lstm_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, cnn_feat, lstm_feat):
        return self.fc(torch.cat([cnn_feat, lstm_feat], dim=1))


class HybridSER(nn.Module):
    def __init__(self, n_mels=64, n_mfcc=40, num_classes=7):
        super().__init__()
        self.cnn        = CNN_TFA(n_mels=n_mels, out_dim=256)
        self.lstm       = BiLSTM_A(input_dim=n_mfcc, hidden_dim=128, out_dim=256, num_layers=1)
        self.classifier = FusionNet(cnn_dim=256, lstm_dim=256, hidden=256, num_classes=num_classes)

    def forward(self, mel, mfcc):
        return self.classifier(self.cnn(mel), self.lstm(mfcc))