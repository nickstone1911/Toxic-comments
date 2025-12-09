# Toxic Comment Classification with Deep Learning

This project builds and compares deep learning models to classify toxic comments in online Wikipedia discussions. The goal is to predict six types of toxic behavior:

- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

The notebook demonstrates a full deep learning workflow: data provenance, EDA, preprocessing, baseline modeling, LSTM sequence modeling, evaluation, and discussion.

---

## 📁 Project Structure

toxic-comments/  
│  
├── data/  
│   └── raw/               (train.csv, test.csv)  
│  
├── notebooks/  
│   └── 01_toxic_comments_DL.ipynb  
│  
├── models/                (optional saved models)  
│  
├── reports/  
│   └── figures/           (EDA & training plots)  
│  
└── README.md  

---

## 📊 1. Dataset & Provenance

- **Source:** Jigsaw Toxic Comment Classification Challenge (Kaggle)  
- **Description:** 150,000+ Wikipedia talk-page comments labeled by human raters for multiple toxicity subtypes.  
- **Task:** Multi-label text classification  
- **Labels:** toxic, severe_toxic, obscene, threat, insult, identity_hate  
- **Provenance notes:**  
  - Comments come from real Wikipedia discussion pages.  
  - Labels were assigned by multiple human raters.  
  - Dataset may contain biases — especially around identity terms.  

---

## 🔍 2. Exploratory Data Analysis (EDA)

Key EDA steps performed:

- Examined dataset structure, column types, and missing values  
- Analyzed label imbalance (most comments are non-toxic)  
- Visualized comment length distributions  
- Explored most common words in toxic vs. non-toxic comments  
- Calculated correlations between toxicity labels (e.g., toxic ↔ insult)  
- Identified preprocessing needs: lowercasing, stripping whitespace, tokenization, padding  

---

## 🧹 3. Preprocessing

- Lowercased text  
- Removed empty comments  
- Created a `clean_text` column  
- Prepared labels as 6-dimensional binary vectors  
- Two paths prepared for modeling:  
  1. **TF-IDF features** for the baseline  
  2. **Tokenized + padded sequences** (max length 200, vocab size 50,000) for the LSTM  

---

## 🧠 4. Modeling Approaches

### **4.1 Baseline: TF-IDF + Dense Neural Network**

- High-dimensional TF-IDF vectors (50,000 features)  
- Dense hidden layer + dropout  
- Sigmoid output (multi-label)  
- **Validation results:**  
  - AUC ≈ **0.94**  
  - Binary accuracy ≈ **0.98**  

This set a strong baseline.

---

### **4.2 LSTM Sequence Model**

- Tokenized comments into integer sequences  
- Embedding layer (128-dim)  
- LSTM layer (128 units)  
- Sigmoid output  
- **Validation results:**  
  - AUC ≈ **0.91**  
  - Binary accuracy ≈ **0.97**  

The LSTM performed well but **did not surpass** the baseline without further tuning.

---

## 📈 5. Results & Comparison

| Model | Validation AUC | Validation Accuracy |
|-------|----------------|---------------------|
| TF-IDF + Dense | ~0.94 | ~0.98 |
| LSTM | ~0.91 | ~0.97 |

**Interpretation:**

- TF-IDF captures strong lexical signals (profanity, slurs) that dominate toxicity detection.  
- The LSTM model likely needs more capacity or epochs to outperform the sparse lexical baseline.  
- Sequence models shine more with nuanced or contextual tasks — here, keywords are highly predictive.  

---

## ⚠️ 6. Limitations

- Rare labels (threat, identity_hate) are hard to learn  
- No hyperparameter tuning beyond default settings  
- Only a single LSTM architecture tested  
- Transformers (BERT, DistilBERT) would likely outperform both models  
- Potential demographic bias in toxicity labeling  

---

## 🚀 7. Future Work

Suggested improvements:

- Use **BiLSTM**, **GRU**, or **CNN** architectures  
- Train **Transformer-based** models (BERT/DistilBERT)  
- Use **class weighting** or **focal loss** for rare labels  
- Increase training epochs to 10–20  
- Apply **subword tokenization** (BPE / WordPiece)  
- Conduct fairness analysis across identity-related terms  

---

## ✅ 8. Conclusion

This project demonstrates that:

- Simple lexical features (TF-IDF) can be extremely strong for toxicity classification.  
- Deep sequence models do not automatically outperform baselines without careful tuning.  
- Understanding your dataset, label imbalance, and linguistic patterns is essential before choosing a model.  

The notebook provides a complete end-to-end deep learning workflow and a strong foundation for more advanced experimentation.