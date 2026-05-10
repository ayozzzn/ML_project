import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("../processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

df = pd.read_csv(PROCESSED_DIR / "features_ml.csv")
print(f"Loaded dataset: {df.shape}")
print(df["emotion"].value_counts())

feature_cols = [c for c in df.columns if c not in ["emotion", "language", "dataset"]]
X = df[feature_cols].values.astype(np.float32)
y_raw = df["emotion"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
NUM_CLASSES = len(label_encoder.classes_)
print(f"\nClasses: {label_encoder.classes_}")
print(f"Num classes: {NUM_CLASSES}")

scaler = StandardScaler()
X = scaler.fit_transform(X)

import joblib
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

def make_weighted_sampler(y_labels):
    class_counts = np.bincount(y_labels)
    weights = 1.0 / class_counts[y_labels]
    return WeightedRandomSampler(weights, len(weights))

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EmotionDataset(X_train, y_train)
val_dataset   = EmotionDataset(X_val, y_val)
test_dataset  = EmotionDataset(X_test, y_test)

sampler = make_weighted_sampler(y_train)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=64)
test_loader  = DataLoader(test_dataset, batch_size=64)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(input_size // 2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = AttentionPooling(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

INPUT_SIZE = X_train.shape[1]
model = CNN_LSTM(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  label_smoothing=self.label_smoothing,
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class_weight_path = PROCESSED_DIR / "class_weights.json"
class_weights_tensor = None
if class_weight_path.exists():
    with open(class_weight_path) as f:
        class_weights_dict = json.load(f)
    weights_list = [class_weights_dict.get(cls, 1.0) for cls in label_encoder.classes_]
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)
    print(f"\n⚖️  Using class weights: {dict(zip(label_encoder.classes_, weights_list))}")

criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor, label_smoothing=0.1)

EPOCHS = 60
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

def warmup_cosine_schedule(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            preds = out.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_f1

print("\n🚀 Training CNN+LSTM...\n")
best_val_f1 = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, train_f1 = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_f1 = run_epoch(val_loader, train=False)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODELS_DIR / "cnn_lstm_best.pt")

    if epoch % 5 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | "
              f"LR: {lr_now:.6f}")

print(f"\n✅ Best val F1: {best_val_f1:.3f}")

model.load_state_dict(torch.load(MODELS_DIR / "cnn_lstm_best.pt", map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        out = model(X_batch.to(DEVICE))
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average="macro")

print(f"\n📊 TEST RESULTS")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro F1: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history["train_loss"], label="Train", color="steelblue")
axes[0, 0].plot(history["val_loss"], label="Val", color="coral")
axes[0, 0].set_title("Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history["train_acc"], label="Train", color="steelblue")
axes[0, 1].plot(history["val_acc"], label="Val", color="coral")
axes[0, 1].set_title("Accuracy")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history["val_f1"], label="Val F1", color="green")
axes[1, 0].axhline(y=best_val_f1, color="gray", linestyle="--", alpha=0.5, label=f"Best F1={best_val_f1:.3f}")
axes[1, 0].set_title("Validation Macro F1")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=axes[1, 1])
axes[1, 1].set_title("Confusion Matrix (normalized)")
axes[1, 1].set_ylabel("True")
axes[1, 1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(MODELS_DIR / "cnn_lstm_results.png", dpi=150)
plt.show()
print("💾 Saved results plot (loss + accuracy + F1 + confusion matrix)")