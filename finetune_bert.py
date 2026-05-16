import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# ─── Config ───────────────────────────────────────────────────────────────────
CSV_PATH = "interview_sentiment_dataset.csv"
MODEL_SAVE_PATH = "./interview_sentiment_model"
BASE_MODEL = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─── Dataset ──────────────────────────────────────────────────────────────────
class InterviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)} | Positive: {df['label'].sum()} | Negative: {(df['label']==0).sum()}")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.15,
    random_state=SEED,
    stratify=df["label"].tolist()
)

print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")


# ─── Tokenizer & Model ────────────────────────────────────────────────────────
print("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
model = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
model = model.to(device)

train_dataset = InterviewDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = InterviewDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ─── Optimizer & Scheduler ────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
num_training_steps = EPOCHS * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return preds, true_labels


print("\nStarting training...")
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    preds, true_labels = evaluate(model, val_loader, device)
    acc = accuracy_score(true_labels, preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val Accuracy: {acc:.4f}")

print("\nClassification Report:")
preds, true_labels = evaluate(model, val_loader, device)
print(classification_report(true_labels, preds, target_names=["Negative", "Positive"]))


# ─── Save model ───────────────────────────────────────────────────────────────
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")


# ─── Quick inference test ─────────────────────────────────────────────────────
def predict(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label = "Positive" if probs[1] > probs[0] else "Negative"
    confidence = max(probs)
    return label, confidence

print("\nInference test:")
test_answers = [
    "I have strong experience and I'm very excited about this opportunity.",
    "I don't really know why I applied. I just need a job."
]
for ans in test_answers:
    label, conf = predict(ans, model, tokenizer, device)
    print(f"  '{ans[:60]}...' → {label} ({conf:.2%})")
