# 🚀 Fine-Tuning Inefficiency: Parameter-Efficient BERT Training

## 📌 Overview
This project explores the inefficiency of full fine-tuning in large language models like BERT and implements **parameter-efficient fine-tuning (PEFT)** methods along with **partial fine-tuning (Attention-only)** to reduce computational cost while maintaining performance.

We compare:
- Full Fine-Tuning (Baseline BERT)
- Attention-Only Training (Partial Fine-Tuning)
- LoRA (Low-Rank Adaptation)
- Adapter-based Fine-Tuning (IA3 using PEFT)

---

## 🎯 Objective
Full fine-tuning of BERT is computationally expensive due to training all parameters (~109M).  
This project aims to:
- Reduce trainable parameters  
- Reduce training time  
- Maintain comparable performance  

---

## 📂 Project Structure

```
bert-efficient-ft/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── baseline/
│   ├── attention/
│   ├── lora/
│   ├── adapter/
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_attention.py
│   ├── train_lora.py
│   ├── train_adapter.py
│
├── results/
│   ├── metrics.csv
│   ├── training_time.png
│   ├── plot_results.py
│   ├── trainable_params.png
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset
- Dataset: **SQuAD (Stanford Question Answering Dataset)**
- Subset used:
  - Train: 1000 samples  
  - Validation: 200 samples  

---

## ⚙️ Installation

```bash
git clone https://github.com/KaustavModak/Fine-Tuning-Inefficiency.git
cd Fine-Tuning-Inefficiency

pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Load Data
```bash
python src/data_loader.py
```

### 2. Preprocess Data
```bash
python src/preprocess.py
```

### 3. Train Models

#### Baseline Model
```bash
python src/train_baseline.py
```

#### Attention-Only Model
```bash
python src/train_attention.py
```

#### LoRA Model
```bash
python src/train_lora.py
```

#### Adapter Model (IA3)
```bash
python src/train_adapter.py
```

---

## 📈 Results

| Model | Accuracy | Training Time (sec) | Trainable Params | % Trainable |
|------|----------|--------------------|------------------|------------|
| Baseline BERT | 0.09 | 5124.92 | 109M | 100% |
| Attention-Only | ~0.045 | 2068.53 | ~28M | ~26% |
| LoRA | ~0.01 | 2174.67 | 294,912 | 0.27% |
| Adapter (IA3) | ~0.01 | 2245.16 | 18,432 | 0.0169% |

---

## 📊 Visualizations

### Training Time Comparison
![Training Time](results/training_time.png)

### Trainable Parameters Comparison
![Parameters](results/trainable_params.png)

---

## 🔍 Key Insights

### Baseline BERT
- High computation cost  
- Slow training (~85 minutes)  
- Trains all parameters  

### Attention-Only Training
- Reduces parameters to ~26%  
- Much faster than baseline  
- Better performance than PEFT methods  
- Strong middle-ground approach  

### LoRA
- ~2× faster training  
- Only 0.27% parameters updated  
- Efficient and scalable  

### Adapter (IA3)
- Most parameter-efficient  
- Only 0.0169% parameters trained  
- Suitable for low-resource environments  

---

## 🧠 Conclusion

Parameter-efficient methods like LoRA and Adapter-based fine-tuning significantly reduce computational cost while maintaining reasonable performance.

Additionally, training only attention layers provides a strong intermediate approach that balances efficiency and accuracy.

These methods are ideal for:
- Low-resource environments  
- Faster experimentation  
- Scalable machine learning systems  

---

## 🛠️ Tech Stack

- Python  
- Hugging Face Transformers  
- PEFT (LoRA, IA3)  
- Datasets Library  
- PyTorch  
- Matplotlib  

---

## 📌 Future Improvements

- Train on full dataset  
- Use better evaluation metrics (F1 / Exact Match)  
- Hyperparameter tuning  
- GPU acceleration  
- Compare more partial fine-tuning strategies  

---

## 👨‍💻 Author

Kaustav Modak  

---

## ⭐ Acknowledgements

- Hugging Face  
- SQuAD Dataset  
- Research work on LoRA & Adapter-based fine-tuning  