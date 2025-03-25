from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load Facebook RoBERTa hate speech model and tokenizer
MODEL_NAME = "facebook/roberta-hate-speech-dynabench-r4-target"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

def classify_text(text):
    """Classifies text for hate speech using the Facebook RoBERTa model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    labels = ["nothate", "hate"]  # Adjust based on actual model labels
    result = {labels[i]: float(scores[i]) for i in range(len(labels))}
    return result