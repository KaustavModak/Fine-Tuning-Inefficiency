# 🚀 BERT Efficient Fine-Tuning (LoRA & Adapters)

## 📌 Overview

This project explores **efficient fine-tuning techniques for BERT** on the **SQuAD Question Answering dataset**.

Instead of training all parameters (which is computationally expensive), we compare:

* ✅ Full Fine-Tuning (Baseline)
* ⚡ LoRA (Low-Rank Adaptation)
* 🔌 Adapter Layers

---

## 🎯 Objective

> Reduce computational cost while maintaining performance.

We evaluate:

* Accuracy / F1 Score
* Training Time
* Number of Trainable Parameters
* Model Size

---

## 📂 Project Structure

```
bert-efficient-ft/
│
├── data/
│   ├── raw/                  # Raw dataset (SQuAD subset)
│   ├── processed/            # Tokenized dataset
│
├── models/
│   ├── baseline/
│   ├── lora/
│   ├── adapter/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_lora.py
│   ├── train_adapter.py
│   ├── evaluate.py
│   ├── utils.py
│
├── results/
│   ├── metrics.csv
│   ├── plots/
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/KaustavModak/Fine-Tuning-Inefficiency.git
cd Fine-Tuning-Inefficiency
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Download Dataset

```bash
python src/data_loader.py
```

---

### 4️⃣ Preprocess Data

```bash
python src/preprocess.py
```

---

## 🧠 Training

### 🔹 Baseline (Full Fine-Tuning)

```bash
python src/train_baseline.py
```

---

### 🔹 LoRA (Efficient Fine-Tuning)

```bash
python src/train_lora.py
```

---

### 🔹 Adapter Layers

```bash
python src/train_adapter.py
```

---

## 📊 Evaluation

```bash
python src/evaluate.py
```

---

## 📈 Results

| Model         | Accuracy | F1 Score | Training Time | Trainable Params |
| ------------- | -------- | -------- | ------------- | ---------------- |
| Baseline BERT | TBD      | TBD      | TBD           | ~110M            |
| LoRA          | TBD      | TBD      | TBD           | ↓↓↓              |
| Adapter       | TBD      | TBD      | TBD           | ↓↓↓              |

---

## ⚡ Key Insights

* Full fine-tuning is expensive but performs best.
* LoRA drastically reduces trainable parameters.
* Adapters provide a balance between efficiency and performance.

---

## 🛠️ Tech Stack

* Python
* HuggingFace Transformers
* Datasets
* PyTorch

---

## 📌 Future Work

* Layer freezing strategies
* Hybrid LoRA + Adapter models
* Hyperparameter tuning

---

## 👨‍💻 Author

**Kaustav Modak**

---

## ⭐ If you found this helpful

Give this repo a star ⭐
