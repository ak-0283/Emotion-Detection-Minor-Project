## 🎭 Emotion Detection 

<!-- ![Emotion Detection Banner](https://github.com/yourusername/Emotion-Detection-Minor-Project/assets/banner-placeholder.png) -->

> **A real-time facial emotion recognition web app built with TensorFlow, Keras, and Streamlit.**

This project detects human emotions such as 😄 **happy**, 😡 **angry**, 😢 **sad**, 😨 **fear**, 😕 **disgust**, 😮 **surprise**, and 😐 **neutral** from facial images or live webcam feeds — using **six deep learning models** trained on the FER-2013 dataset.

> 🎓 This repository is developed as a **BCA 5th Semester Major Project**.

---

## 🚀 Features

✅ **Six powerful CNN-based models:**

* 🧠 **VGG16** — Transfer learning on RGB faces
* ⚡ **ResNet50** — Robust residual feature extraction
* 🔍 **InceptionV3** — Multi-scale feature detection
* 🌿 **EfficientNetB0** — Lightweight and efficient grayscale model
* 🧩 **HybridNet** — Custom fusion model trained on **grayscale** images
* 🧱 **CustomCNN** — Designed and trained from scratch

✅ **Two prediction modes:**

* 🖼️ Upload an image and detect emotion
* 📷 Real-time emotion detection using your webcam

✅ **Smart ensemble voting:**
Combines predictions from multiple models to form a final consensus result.

✅ **User-friendly Streamlit interface**
Clean, responsive, and interactive dashboard.

---

## 🧰 Tech Stack

| Category             | Tools / Libraries                                |
| -------------------- | ------------------------------------------------ |
| **Frontend**         | 🧩 Streamlit                                     |
| **Backend / ML**     | TensorFlow, Keras, NumPy, OpenCV                 |
| **Image Processing** | Pillow, OpenCV, NumPy                            |
| **Dataset**          | FER-2013 (Facial Expression Recognition Dataset) |
| **Visualization**    | Streamlit Charts, Emojis 😄😡😢                  |

---

## 🏗️ Project Structure

```
Emotion-Detection-Minor-Project/
│
├── app.py                         # Streamlit app (main file)
│
├── models/                        # Pretrained and custom models
│   ├── VGG16_emotion_model.keras
│   ├── ResNet50_emotion_model.keras
│   ├── InceptionV3_emotion_model.keras
│   ├── EfficientNetB0_emotion_model.keras
│   ├── hybrid/
│   │   └── HybridNet_emotion_model_FIXED.keras
│   └── custom/
│       └── CustomCNN_emotion_model.keras
│
├── google_collab/
│   ├── pretrained/
│   │   └── pretrained.ipynb
│   └── custom/
│       └── custombuilt.ipynb
│
├── requirements.txt               # Dependencies
├── README.md                      # Documentation (this file)
└── assets/                        # Images / screenshots / banner (optional)
```

---

## ⚙️ Installation Guide

### 🔹 1. Clone the repository

```bash
git clone https://github.com/yourusername/Emotion-Detection-Minor-Project.git
cd Emotion-Detection-Minor-Project
```

### 🔹 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

### 🔹 3. Install dependencies

```bash
pip install -r requirements.txt

```

**📌 till now i have not uploaded the requirements.txt i will upload.**

### 🔹 4. Run the app

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Face Detection** — Detects faces using OpenCV’s Haar cascades
2. **Preprocessing** — Crops, resizes, and normalizes faces for each model
3. **Model Prediction** — Passes images to CNN models for classification
4. **Ensemble Fusion** — Uses majority voting to determine final emotion
5. **Visualization** — Displays predictions and confidence scores in Streamlit

---

## 🎯 Supported Emotions

| Emotion | Emoji    | Description                 |
| ------- | -------- | --------------------------- |
| 😄      | Happy    | Joy, amusement              |
| 😢      | Sad      | Unhappiness, disappointment |
| 😡      | Angry    | Rage, frustration           |
| 😨      | Fear     | Anxiety, nervousness        |
| 😕      | Disgust  | Aversion, dislike           |
| 😮      | Surprise | Shock, astonishment         |
| 😐      | Neutral  | Calm, blank expression      |

---

## 🧩 HybridNet Model (Custom Fusion)

> 🧠 A hybrid model combining EfficientNetB0 and custom CNN layers, trained on grayscale FER-2013 images for lightweight, high-accuracy performance.

Key components:

* `Lambda(grayscale_to_rgb)` layer for safe conversion
* Pretrained EfficientNet backbone
* Custom dense head for fine-tuning
* Fixed to deserialize safely across environments

---

## 🖼️ Screenshots

| Upload Mode                                                                                          | Live Webcam Mode                                                                                     |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| ![Upload](https://github.com/yourusername/Emotion-Detection-Minor-Project/assets/upload-preview.png) | ![Webcam](https://github.com/yourusername/Emotion-Detection-Minor-Project/assets/webcam-preview.png) |

---

## 💡 Example Output

```
📦 Model Loading Status
✅ Loaded VGG16 successfully! (RGB mode)
✅ Loaded ResNet50 successfully! (RGB mode)
✅ Loaded InceptionV3 successfully! (RGB mode)
✅ Loaded EfficientNetB0 successfully! (Grayscale mode)
✅ Loaded HybridNet successfully! (Grayscale mode) 📌hybrid is still not working.
✅ Loaded CustomCNN successfully! (RGB mode)
🎉 Successfully loaded 6 models!
```

**Prediction Example:**

```
🧠 HybridNet → Happy (98.23%)
⚡ ResNet50 → Happy (96.81%)
🌿 EfficientNetB0 → Happy (95.12%)

🧩 Final Consensus: Happy 😄
```

---

## 📦 Requirements

```
streamlit
tensorflow
keras
opencv-python
pillow
numpy
pickle-mixin
```

---

## 👨‍💻 Developer

**Your Name**
📧 [Gmail](mailto:abhay.kr2803@gmail.com)
💼 [LinkedIn](https://www.linkedin.com/in/abhay-kumar-117b4327b/)
🌐 [GitHub](https://github.com/ak-0283)

### 🤝 Project Credit

* **Friend Contributor**
* GitHub: [@friend-github-id](https://github.com/Suraj-chetri08)

<div align="center">
  <h2 style="font-size:3rem;">Our Contributors <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Red%20Heart.png" alt="Red Heart" width="40" height="40" /></h2>
  <h3>Thank you for contributing to our repository</h3>

<a href="https://github.com/ak-0283/Emotion-Detection-Minor-Project/graphs/contributors">
<img src="https://contributors-img.web.app/image?repo=ak-0283/Emotion-Detection-Minor-Project"/>
  
  </a>
<p style="font-family:var(--ff-philosopher);font-size:3rem;"><b> Show some <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Red%20Heart.png" alt="Red Heart" width="40" height="40" /> by starring this awesome repository!

</div>


> *“Machines that understand emotions bring us one step closer to empathetic AI.”* 🤖💙

---

## ⭐ Show Your Support

If you like this project:

* Give it a ⭐ on GitHub
* Share it with your friends or classmates
* Tag me on LinkedIn if you use it in your portfolio!

---

## 🏁 License

This project is released under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

### 🧠 Inspiration

Built as part of a **Minor Project in Deep Learning** — focused on combining transfer learning with custom CNNs for emotion recognition from facial features.

---
