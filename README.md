# Emotion Classification using DistilBERT 

## 1. Project Overview
This project implements an emotion classification system that detects human emotions from text.  
Unlike traditional sentiment analysis (positive / negative / neutral), the system classifies text into **six distinct emotions**:

- Joy  
- Sadness  
- Anger  
- Fear  
- Love  
- Surprise  

### Importance of Emotion Detection
Emotion classification has many real-world applications, including:
- Mental health monitoring  
- Improving customer support systems   
- Social media trend analysis  
- Enabling empathetic human–AI interaction   

---

## 2. Problem Statement
Manually identifying emotions in text is challenging due to:
- Diverse linguistic styles used to express emotions  
- Multiple expressions representing the same emotion  
- Subtle emotions such as *love* and *surprise* being hard to detect  
- Massive volumes of online text that make manual analysis impractical  

**Goal:**  
Develop an automated emotion classification system that accurately identifies emotions to support applications in mental health, customer service, and empathetic AI.

---

## 3. Dataset
We used the **`dair-ai/emotion`** dataset from HuggingFace.

- **Emotion Classes:** joy, sadness, anger, fear, love, surprise  
- **Samples:** ~20,000 labeled sentences  
- **Splits:** Training / Validation / Test  

### Why This Dataset?
- Clean and well-structured  
- Balanced across emotion classes  
- Suitable for NLP model training  
- Widely used in academic research  

---

## 4. Data Preprocessing
The following preprocessing steps were applied before model training:

- Tokenization using the DistilBERT tokenizer  
- Padding and truncation to a maximum sequence length of 128  
- Conversion of labels to PyTorch tensors  
- Renaming `label` to `labels` for compatibility with the HuggingFace Trainer API  

### Why Tokenization?
- Converts text into numerical representations  
- Generates attention masks  
- Ensures consistent input length for the model  

---

## 5. Model Architecture

### Why DistilBERT?
- A smaller and faster version of BERT (≈40% fewer parameters)  
- Approximately 60% faster training  
- Retains around 97% of BERT’s performance  
- Well-suited for text classification tasks  

### Fine-Tuning Approach
- Add a classification head on top of DistilBERT  
- Fine-tune the last layers and classifier  
- Use cross-entropy loss for 6-class classification  

### Model Workflow
1. **Text Input:** A sentence is provided  
2. **Tokenization:** Converts text into `input_ids` and `attention_mask`  
3. **DistilBERT Encoder:** Generates contextual embeddings  
4. **Classification Layer:** Predicts one of the six emotions  
5. **Softmax:** Selects the emotion with the highest probability  

---

## 6. Training Setup
- **Batch size:** 16  
- **Learning rate:** 2e-5  
- **Epochs:** 3  
- **Weight decay:** 0.01  
- **Evaluation:** During training  
- **Logging & checkpoints:** Disabled  

### Evaluation Metrics
- Accuracy  
- Macro F1-score (effective for handling class imbalance)  

---

## 7. Evaluation & Results
- **Accuracy:** 0.93  
- **Macro F1-score:** 0.88  

### Observations
- Best predicted emotions: **Sadness, Joy**  
- More challenging emotions: **Surprise, Love**  
- Some **Anger** samples were misclassified as **Fear**

---

## 8. Analysis

### Strengths
- High accuracy for common emotions  
- Effectively captures strong emotional cues  

### Weaknesses
- Subtle emotions such as *Surprise* and *Love* are harder to classify  
- Less frequent classes show lower performance  

---

## 9. Conclusion
The fine-tuned DistilBERT model demonstrates strong performance in text-based emotion classification:

- Accurately predicts common emotions like Sadness and Joy  
- Handles subtle emotions reasonably well  
- Suitable for applications in mental health, customer support, social media analysis, and empathetic AI systems  

---

## 10. How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name_or_path = "your-fine-tuned-model"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

while True:
    text = input("Enter a sentence (or 'quit' to exit): ")
    if text.lower() == "quit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        emotion = label_names[pred_idx]

    print(f"Predicted emotion: {emotion}\n")

---

## 11. Requirements
- Python 3.9+
- transformers
- torch
- datasets
- numpy
- scikit-learn
