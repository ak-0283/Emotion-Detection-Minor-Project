# 😃 Facial Emotion Recognition (FER) Using Deep Learning

## 📌 Overview

This project implements **Facial Emotion Recognition (FER)** using **Deep Learning** and **Computer Vision** techniques.
We explore multiple datasets, preprocessing strategies, model architectures, transfer learning, and statistical validation to build a robust FER system.

---

## 🧩 System Flow Architecture

```
Input Image/Video  
   ↓  
Preprocessing (Face Detection, Alignment, Normalization, Augmentation)  
   ↓  
Model Training / Inference (CNN / ResNet / EfficientNet / ViT / CNN+LSTM)  
   ↓  
Post-Processing & Evaluation (Metrics, Confusion Matrix, Statistical Tests)  
   ↓  
Output: Predicted Emotion + Reports
```

---

## 📂 Datasets Used

* 📊 **FER2013** (Kaggle)
* 😊 **RAF-DB** (Real-world Affective Face Database)
* 🎭 **CK+** (Cohn-Kanade)
* 😐 **JAFFE** (Japanese Female Facial Expression)
* 🎥 **MMI** Facial Expression Database
* 🌍 **AffectNet**

---

## 🛠 Preprocessing Steps

* 👤 Face Detection & Alignment → **MTCNN, Haar Cascades, Dlib**
* 🌑 Grayscale Conversion
* ⚖️ Normalization (0–1 or -1–1)
* 🎨 Data Augmentation (flip, rotation, brightness, zoom)

---

## 🏗 Model Architectures

### 🔹 Custom Models

* Simple CNN (3–5 layers)
* Deep CNN (ResNet-style with skip connections)

### 🔹 Hybrid Models

* CNN + LSTM/GRU (spatio-temporal)
* CNN + Attention (CBAM / Self-Attention)
* Multimodal Fusion (Image + Video)

### 🔹 Pretrained (Transfer Learning)

* 🐶 VGG16
* 🔗 ResNet50
* 🔎 InceptionV3
* ⚡ EfficientNet
* 🧠 Vision Transformer (ViT)

---

## ⚙️ Hyperparameter Optimization

* 🔍 Grid Search
* 🎲 Random Search
* 🧪 Bayesian Optimization (Optuna, Hyperopt)
* 📊 Tuned parameters: Learning rate, batch size, dropout, optimizer, loss function

---

## 🧪 Ablation Study

* ❌ Without augmentation
* ❌ Without dropout / batch normalization
* ❌ Without attention
* ❌ Without transfer learning
* 📹 With vs. Without temporal modeling

---

## 📈 Evaluation Metrics

* ✅ Accuracy
* 📊 Precision, Recall, F1-Score
* 🔀 Confusion Matrix
* 🎯 AUC-ROC
* 📏 Mean Absolute Error (MAE)
* 🤝 Cohen’s Kappa

---

## 📊 Statistical Validation

* 📌 Paired t-test / Wilcoxon signed-rank test
* 📌 ANOVA (multiple model comparison)
* 📌 Confidence Intervals

---

## 🔥 Explainability

* 🖼 **Grad-CAM** visualizations to see what the model is focusing on

---

## 💻 Tech Stack

* 🐍 Python 3.x
* 🔥 PyTorch (Deep Learning)
* 👁 OpenCV (Computer Vision)
* 🤗 HuggingFace / timm (Pretrained models)
* 📊 NumPy, Pandas, Matplotlib, Seaborn (Data & Visualization)
* ⚙️ Albumentations (Augmentation)
* 📈 Scikit-learn, SciPy (Metrics & Statistics)

---

## 🚀 How to Run

```bash
# 1. Clone repo
git clone https://github.com/yourusername/FER-DeepLearning.git
cd FER-DeepLearning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess dataset (face detection, alignment)
python src/preprocess.py

# 4. Train model
python src/train.py --data data_processed/FER2013_faces --model resnet50 --epochs 30

# 5. Evaluate across datasets
python src/evaluate.py --model_path best_model.pth --source FER2013 --target CK_plus
```

---

<!-- ## 🎯 Results (Sample)

| Model        | Dataset | Accuracy | F1-Score |
| ------------ | ------- | -------- | -------- |
| ResNet50     | FER2013 | 72%      | 0.70     |
| EfficientNet | RAF-DB  | 75%      | 0.73     |
| ViT          | CK+     | 92%      | 0.91     |

---
-->

## 📌 Future Work

* 🔮 Real-time FER via webcam (OpenCV + PyTorch)
* 📹 Video-based FER with LSTM/GRU
* ☁️ Deploy as Web App (Flask / FastAPI / Streamlit)
* 🌐 Browser inference (TensorFlow.js)

---

<!-- ## 🙌 Contributors

👨‍💻 **Your Name** – Research, Implementation, Documentation

--- 
-->

## ⭐ Show Your Support

If you like this project, please ⭐ the repo and share it!

---
