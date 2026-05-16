import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "./interview_sentiment_model"
MAX_LEN = 128

class SentimentAnalyser:
    def __init__(self):
        print("Loading fine-tuned sentiment model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded.")

    def predict(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        label = "Positive" if probs[1] > probs[0] else "Negative"
        return {
            "label": label,
            "confidence": float(max(probs)),
            "positive_score": float(probs[1]),
            "negative_score": float(probs[0])
        }


# ── Drop-in usage in your Tkinter app ─────────────────────────────────────────
# from sentiment import SentimentAnalyser
# analyser = SentimentAnalyser()
# result = analyser.predict(user_answer)
# print(result)
# → {'label': 'Positive', 'confidence': 0.94, 'positive_score': 0.94, 'negative_score': 0.06}
