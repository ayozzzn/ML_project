import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    get_cosine_schedule_with_warmup
)
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("../processed")
DATA_DIR = Path("../data")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
MAX_DURATION = 5
BATCH_SIZE = 8
EPOCHS = 10
LR = 5e-5
GRAD_ACCUM_STEPS = 4

import io, joblib

df = pd.read_csv(PROCESSED_DIR / "features_full.csv")
df = df[["filepath", "emotion", "language", "dataset"]].dropna()
print(f"Total dataset size: {len(df)}")
print(df["emotion"].value_counts())
print(df["dataset"].value_counts())

RESD_DIR = DATA_DIR / "RESD"
resd_audio_cache = {}
parquet_files = list(RESD_DIR.rglob("*.parquet")) if RESD_DIR.exists() else []

if parquet_files:
    print(f"\n⬇️  Pre-loading RESD audio from {len(parquet_files)} parquet file(s)...")
    global_idx = 0
    for pf in parquet_files:
        try:
            df_pq = pd.read_parquet(pf)
        except Exception as e:
            print(f"  ❌ {pf.name}: {e}")
            continue
        if 'speech' not in df_pq.columns:
            continue
        for idx, row in df_pq.iterrows():
            key = f"resd_parquet_{global_idx}"
            global_idx += 1
            try:
                audio_data = row['speech']
                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get('bytes')
                    if audio_bytes is None:
                        continue
                    with io.BytesIO(audio_bytes) as f:
                        y, orig_sr = sf.read(f, dtype='float32')
                elif isinstance(audio_data, bytes):
                    with io.BytesIO(audio_data) as f:
                        y, orig_sr = sf.read(f, dtype='float32')
                else:
                    continue
                if len(y.shape) > 1:
                    y = y.mean(axis=1)
                if orig_sr != SAMPLE_RATE:
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
                if len(y) < SAMPLE_RATE * 0.5:
                    continue
                resd_audio_cache[key] = y
            except Exception:
                continue
    print(f"  ✅ RESD cache: {len(resd_audio_cache)} waveforms loaded")
else:
    print("⚠️  No RESD parquet files found")

label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["emotion"])
NUM_CLASSES = len(label_encoder.classes_)
id2label = {i: l for i, l in enumerate(label_encoder.classes_)}
label2id = {l: i for i, l in id2label.items()}
print(f"\nClasses ({NUM_CLASSES}): {id2label}")
joblib.dump(label_encoder, MODELS_DIR / "label_encoder_wav2vec.pkl")

train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label_id"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label_id"], random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def spec_augment_waveform(y, sr=SAMPLE_RATE, max_mask_pct=0.15, num_masks=2):
    y = y.copy()
    n = len(y)
    for _ in range(num_masks):
        mask_len = int(n * np.random.uniform(0, max_mask_pct))
        mask_start = np.random.randint(0, n - mask_len)
        y[mask_start:mask_start + mask_len] = 0
    return y

class SpeechEmotionDataset(Dataset):
    def __init__(self, df, processor, max_duration=MAX_DURATION, sr=SAMPLE_RATE, augment=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_len = max_duration * sr
        self.sr = sr
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = str(row["filepath"])
        y = None

        if fp.startswith("resd_parquet_"):
            y = resd_audio_cache.get(fp)
            if y is None:
                y = np.zeros(self.sr, dtype=np.float32)

        if y is None:
            try:
                y, _ = librosa.load(fp, sr=self.sr, mono=True)
            except Exception:
                y = np.zeros(self.sr, dtype=np.float32)

        if len(y) > self.max_len:
            if self.augment:
                start = np.random.randint(0, len(y) - self.max_len)
            else:
                start = (len(y) - self.max_len) // 2
            y = y[start:start + self.max_len]
        else:
            y = np.pad(y, (0, self.max_len - len(y)))

        if self.augment:
            y = spec_augment_waveform(y, sr=self.sr)

        inputs = self.processor(y, sampling_rate=self.sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze(0)
        label = int(row["label_id"])
        return input_values, label

print("\n⬇️  Loading Wav2Vec2 processor & model...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

model.freeze_feature_extractor()

N_UNFREEZE = 4
total_layers = len(model.wav2vec2.encoder.layers)
for i, layer in enumerate(model.wav2vec2.encoder.layers):
    if i >= total_layers - N_UNFREEZE:
        for param in layer.parameters():
            param.requires_grad = True
    else:
        for param in layer.parameters():
            param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,} (top {N_UNFREEZE} transformer layers + classifier)")

model = model.to(DEVICE)

train_ds = SpeechEmotionDataset(train_df, processor, augment=True)
val_ds   = SpeechEmotionDataset(val_df, processor, augment=False)
test_ds  = SpeechEmotionDataset(test_df, processor, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=1e-2
)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
warmup_steps = total_steps // 10
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
best_f1 = 0

print(f"\n🚀 Fine-tuning Wav2Vec2 for {EPOCHS} epochs (grad_accum={GRAD_ACCUM_STEPS})...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    optimizer.zero_grad()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)
        labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)

        outputs = model(inputs, labels=labels)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss += outputs.loss.item()

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {outputs.loss.item():.4f}")

    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels_t = torch.tensor(labels, dtype=torch.long).to(DEVICE)
            outputs = model(inputs, labels=labels_t)
            val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    val_acc = accuracy_score(all_labels, all_preds)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)

    print(f"\n📊 Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        model.save_pretrained(MODELS_DIR / "wav2vec2_emotion")
        processor.save_pretrained(MODELS_DIR / "wav2vec2_emotion")
        print(f"  💾 Saved best model (F1={best_f1:.4f})")

print("\n📊 Loading best model for test evaluation...")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODELS_DIR / "wav2vec2_emotion").to(DEVICE)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = outputs.logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

print(f"\n✅ TEST RESULTS")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro F1: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))

print("\n🌍 Cross-lingual evaluation...")
lang_results = {}

for lang, subset in [("English", test_df[test_df["language"] == "en"]),
                     ("Russian", test_df[test_df["language"] == "ru"])]:
    if len(subset) == 0:
        print(f"  No {lang} test samples found.")
        continue
    ds = SpeechEmotionDataset(subset, processor, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    preds, labels = [], []
    with torch.no_grad():
        for inputs, lbls in loader:
            out = model(inputs.to(DEVICE))
            preds.extend(out.logits.argmax(1).cpu().numpy())
            labels.extend(lbls)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    lang_results[lang] = {"f1": f1, "acc": acc, "n": len(subset)}
    print(f"  {lang}: Accuracy={acc:.4f}, F1={f1:.4f} (n={len(subset)})")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history["train_loss"], label="Train", color="steelblue")
axes[0, 0].plot(history["val_loss"], label="Val", color="coral")
axes[0, 0].set_title("Training & Validation Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history["val_f1"], label="Val F1", color="green")
axes[0, 1].plot(history["val_acc"], label="Val Acc", color="purple")
axes[0, 1].axhline(y=best_f1, color="gray", linestyle="--", alpha=0.5)
axes[0, 1].set_title("Validation F1 & Accuracy")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix (normalized) — Test Set")
axes[1, 0].set_ylabel("True")
axes[1, 0].set_xlabel("Predicted")

if lang_results:
    langs = list(lang_results.keys())
    f1_vals = [lang_results[l]["f1"] for l in langs]
    acc_vals = [lang_results[l]["acc"] for l in langs]
    x = np.arange(len(langs))
    w = 0.35
    axes[1, 1].bar(x - w/2, acc_vals, w, label="Accuracy", color="steelblue", alpha=0.8)
    axes[1, 1].bar(x + w/2, f1_vals, w, label="Macro F1", color="coral", alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(langs)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Cross-lingual Performance")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(MODELS_DIR / "wav2vec2_results.png", dpi=150)
plt.show()
print("💾 Saved results plot")

print("\n✅ Wav2Vec2 training complete!")