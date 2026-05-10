import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("../data")
OUTPUT_DIR = Path("../processed")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
N_MFCC = 40
SEGMENT_DURATION = 3

print("✅ Imports OK")

def parse_ravdess_label(filepath):
    mapping = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }
    parts = Path(filepath).stem.split("-")
    return mapping.get(parts[2], "unknown")

def parse_cremad_label(filepath):
    mapping = {
        "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
        "HAP": "happy", "NEU": "neutral", "SAD": "sad"
    }
    parts = Path(filepath).stem.split("_")
    return mapping.get(parts[2], "unknown")

def parse_tess_label_folder(folder_name):
    part = folder_name.lower().split("_")[-1]
    mapping = {
        "angry": "angry", "disgust": "disgust", "fear": "fearful",
        "happy": "happy", "neutral": "neutral", "ps": "surprise",
        "sad": "sad", "surprise": "surprise"
    }
    return mapping.get(part, "unknown")

def load_and_preprocess(filepath, sr=SAMPLE_RATE):
    try:
        y, orig_sr = librosa.load(filepath, sr=None, mono=True)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        y = nr.reduce_noise(y=y, sr=sr, stationary=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        return y
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def augment_audio(y, sr=SAMPLE_RATE):
    augmented = []

    for rate in [0.9, 1.1]:
        try:
            y_stretched = librosa.effects.time_stretch(y, rate=rate)
            augmented.append(y_stretched)
        except Exception:
            pass

    for steps in [-2, 2]:
        try:
            y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            augmented.append(y_pitched)
        except Exception:
            pass

    noise = np.random.normal(0, 0.002, len(y))
    augmented.append(y + noise)

    return augmented

def extract_features(y, sr=SAMPLE_RATE):
    features = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i}_std"] = np.std(mfcc[i])

    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i in range(N_MFCC):
        features[f"delta_mfcc_{i}_mean"] = np.mean(delta_mfcc[i])
        features[f"delta2_mfcc_{i}_mean"] = np.mean(delta2_mfcc[i])

    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_voiced = f0[f0 > 0]
    features["pitch_mean"] = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0
    features["pitch_std"] = np.std(f0_voiced) if len(f0_voiced) > 0 else 0
    features["pitch_range"] = (np.max(f0_voiced) - np.min(f0_voiced)) if len(f0_voiced) > 0 else 0

    rms = librosa.feature.rms(y=y)[0]
    features["energy_mean"] = np.mean(rms)
    features["energy_std"] = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = np.mean(centroid)
    features["spectral_centroid_std"] = np.std(centroid)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = np.mean(rolloff)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(contrast.shape[0]):
        features[f"spectral_contrast_{i}_mean"] = np.mean(contrast[i])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features["mel_mean"] = np.mean(mel_db)
    features["mel_std"] = np.std(mel_db)

    non_silent = librosa.effects.split(y, top_db=20)
    total_speaking_frames = sum(end - start for start, end in non_silent)
    features["speaking_rate"] = total_speaking_frames / len(y) if len(y) > 0 else 0
    features["pause_ratio"] = 1.0 - features["speaking_rate"]
    features["num_pauses"] = len(non_silent)

    return features

def build_dataset(dataset_name, audio_dir, label_parser, extensions=(".wav", ".mp3"),
                  augment_minority=False, minority_labels=None):
    rows = []
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        print(f"⚠️  {dataset_name} directory not found: {audio_dir}. Skipping.")
        return pd.DataFrame()

    files = [f for f in audio_dir.rglob("*") if f.suffix.lower() in extensions]
    print(f"📂 {dataset_name}: found {len(files)} files")

    for i, filepath in enumerate(files):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(files)}...")

        label = label_parser(filepath)
        if label == "unknown":
            continue

        y = load_and_preprocess(str(filepath))
        if y is None or len(y) < SAMPLE_RATE * 0.5:
            continue

        feats = extract_features(y)
        feats["label"] = label
        feats["dataset"] = dataset_name
        feats["filepath"] = str(filepath)
        feats["language"] = "en"
        rows.append(feats)

        if augment_minority and minority_labels and label in minority_labels:
            for aug_y in augment_audio(y):
                if len(aug_y) < SAMPLE_RATE * 0.5:
                    continue
                aug_feats = extract_features(aug_y)
                aug_feats["label"] = label
                aug_feats["dataset"] = dataset_name + "_aug"
                aug_feats["filepath"] = str(filepath) + "_aug"
                aug_feats["language"] = "en"
                rows.append(aug_feats)

    df = pd.DataFrame(rows)
    print(f"  ✅ {dataset_name}: {len(df)} samples extracted")
    return df

def build_tess(tess_dir):
    rows = []
    tess_dir = Path(tess_dir)
    if not tess_dir.exists():
        print(f"⚠️  TESS directory not found: {tess_dir}. Skipping.")
        return pd.DataFrame()

    for folder in tess_dir.iterdir():
        if not folder.is_dir():
            continue
        label = parse_tess_label_folder(folder.name)
        if label == "unknown":
            continue
        for filepath in folder.glob("*.wav"):
            y = load_and_preprocess(str(filepath))
            if y is None or len(y) < SAMPLE_RATE * 0.5:
                continue
            feats = extract_features(y)
            feats["label"] = label
            feats["dataset"] = "TESS"
            feats["filepath"] = str(filepath)
            feats["language"] = "en"
            rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"  ✅ TESS: {len(df)} samples extracted")
    return df

def build_resd(resd_dir):
    import io
    import soundfile as sf

    resd_dir = Path(resd_dir)
    parquet_files = list(resd_dir.rglob("*.parquet"))

    if not parquet_files:
        print(f"⚠️  No .parquet files found in {resd_dir}. Skipping RESD.")
        return pd.DataFrame()

    print(f"📂 RESD: found {len(parquet_files)} parquet file(s): {[f.name for f in parquet_files]}")

    RESD_EMOTION_MAP = {
        "anger": "angry", "disgust": "disgust", "fear": "fearful",
        "happiness": "happy", "neutral": "neutral", "sadness": "sad",
        "enthusiasm": "happy",
    }

    rows = []
    for parquet_path in parquet_files:
        print(f"  Reading {parquet_path.name}...")
        try:
            df_pq = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  ❌ Could not read {parquet_path.name}: {e}")
            continue

        if 'speech' not in df_pq.columns or 'emotion' not in df_pq.columns:
            print(f"  ❌ Missing required columns")
            continue

        for idx, row in df_pq.iterrows():
            if idx % 100 == 0:
                print(f"    Processing row {idx}/{len(df_pq)}...")

            raw_label = row['emotion'].lower()
            label = RESD_EMOTION_MAP.get(raw_label, "unknown")
            if label == "unknown":
                continue

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
                y = nr.reduce_noise(y=y, sr=SAMPLE_RATE, stationary=True)
                y, _ = librosa.effects.trim(y, top_db=20)
                if len(y) < SAMPLE_RATE * 0.5:
                    continue
            except Exception:
                continue

            feats = extract_features(y)
            feats["label"] = label
            feats["dataset"] = "RESD"
            feats["filepath"] = f"resd_parquet_{idx}"
            feats["language"] = "ru"
            rows.append(feats)

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"  ✅ RESD: {len(df)} samples extracted")
        print(f"  Labels: {df['label'].value_counts().to_dict()}")
    else:
        print("  ⚠️  RESD: 0 samples extracted")
    return df

print("\n🚀 Starting feature extraction...\n")

dfs = []

df_ravdess = build_dataset("RAVDESS", DATA_DIR / "RAVDESS", parse_ravdess_label)
if not df_ravdess.empty:
    dfs.append(df_ravdess)

df_cremad = build_dataset("CREMA-D", DATA_DIR / "CREMA-D", parse_cremad_label)
if not df_cremad.empty:
    dfs.append(df_cremad)

df_tess = build_tess(DATA_DIR / "TESS")
if not df_tess.empty:
    dfs.append(df_tess)

df_resd = build_resd(DATA_DIR / "RESD")
if not df_resd.empty:
    dfs.append(df_resd)

if not dfs:
    print("❌ No datasets found. Make sure your data/ folder has the right structure.")
else:
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Total samples before augmentation: {len(df_all)}")
    print(df_all["label"].value_counts())

LABEL_MAP = {
    "neutral": "neutral", "calm": "calm", "happy": "happy", "sad": "sad",
    "angry": "angry", "fearful": "anxiety", "fear": "anxiety",
    "disgust": "angry", "surprised": "happy", "surprise": "happy",
}

df_all["emotion"] = df_all["label"].map(LABEL_MAP).fillna("neutral")

print("\n🎯 Final emotion distribution:")
emotion_counts = df_all["emotion"].value_counts()
print(emotion_counts)

majority = emotion_counts.max()
minority_classes = emotion_counts[emotion_counts < majority * 0.3].index.tolist()
if minority_classes:
    print(f"\n⚠️  Minority classes (< 30% of majority): {minority_classes}")
    print("   Consider adding augmentation or class weights in training.")

class_weights = {}
total = len(df_all)
for emotion, count in emotion_counts.items():
    class_weights[emotion] = total / (len(emotion_counts) * count)
print(f"\n⚖️  Class weights: {class_weights}")

import json
with open(OUTPUT_DIR / "class_weights.json", "w") as f:
    json.dump(class_weights, f, indent=2)
print(f"💾 Saved class weights: {OUTPUT_DIR / 'class_weights.json'}")

feature_cols = [c for c in df_all.columns if c not in ["label", "dataset", "filepath", "language", "emotion"]]

df_all.to_csv(OUTPUT_DIR / "features_full.csv", index=False)
print(f"\n💾 Saved full dataset: {OUTPUT_DIR / 'features_full.csv'}")

df_ml = df_all[feature_cols + ["emotion", "language", "dataset"]].copy()
df_ml.to_csv(OUTPUT_DIR / "features_ml.csv", index=False)
print(f"💾 Saved ML-ready dataset: {OUTPUT_DIR / 'features_ml.csv'}")

print(f"\n📐 Feature count: {len(feature_cols)} (was ~96 originally, now ~216 with deltas)")
print("\n✅ Preprocessing complete!")