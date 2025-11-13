import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import cv2
from PIL import Image
import os
import keras  # ✅

# ----------------------------------------
# BACKGROUND COLOR (added, UI unchanged)
# ----------------------------------------
def add_bg_color():
    st.markdown("""
        <style>
        .stApp {
            background-color: #090040 !important;
            color: white !important;
        }
        .css-1dp5vir, .stMarkdown, .stText, .stTitle, .stHeader {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

add_bg_color()


# --- Register custom Lambda function used in HybridNet ---
@keras.saving.register_keras_serializable()
def grayscale_to_rgb(img):
    """Custom grayscale → RGB converter for deserialization."""
    return tf.image.grayscale_to_rgb(img)


# ---------------------------
# PAGE CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

# ---------------------------
# LOAD ALL MODELS
# ---------------------------
@st.cache_resource
def load_all_models():
    """Load pretrained, custom, and hybrid CNN models dynamically."""
    model_folders = {
        "VGG16": "models",
        "ResNet50": "models",
        "InceptionV3": "models",
        "EfficientNetB0": "models",
        "HybridNet": "models/hybrid",
        "CustomCNN": "models/custom"
    }

    models, metadata, available = {}, {}, []

    st.subheader("📦 Model Loading Status")

    for name, folder in model_folders.items():
        model_path = os.path.join(folder, f"{name}_emotion_model.keras")
        meta_path = os.path.join(folder, f"{name}_metadata.pkl")

        if name == "HybridNet":
            model_path = os.path.join(folder, "HybridNet_emotion_model_FIXED.keras")

        st.write(f"🔍 **Checking:** `{model_path}`")

        if os.path.exists(model_path):
            try:
                if name == "HybridNet":
                    tf.keras.config.enable_unsafe_deserialization()
                    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                else:
                    model = tf.keras.models.load_model(model_path, compile=False)

                input_shape = model.input_shape

                if os.path.exists(meta_path):
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                else:
                    meta = {"emotion_labels": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}

                models[name] = model
                metadata[name] = meta
                available.append(name)

                if input_shape[-1] == 3:
                    st.success(f"✅ Loaded **{name}** successfully! (RGB mode)")
                elif input_shape[-1] == 1:
                    st.success(f"✅ Loaded **{name}** successfully! (Grayscale mode)")
                else:
                    st.warning(f"⚠️ {name} has unusual input shape: {input_shape}")

                st.info(f"🧩 {name} Input shape: `{input_shape}`")

            except Exception as e:
                st.error(f"❌ Error loading **{name}**: {e}")
        else:
            st.warning(f"⚠️ Missing files for **{name}**. Check your '{folder}' folder.")

    if not available:
        st.error("🚫 No valid models loaded — check your model folders.")
    else:
        st.success(f"🎉 Successfully loaded {len(available)} models: {', '.join(available)}")

    return models, metadata, available


if st.sidebar.button("🔄 Reload Models"):
    st.cache_resource.clear()
    st.experimental_rerun()

models, metadata, available_models = load_all_models()


# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
def preprocess_image(img, model_name="VGG16", expected_channels=3):
    if model_name == "InceptionV3":
        target_size = (299, 299)
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif model_name == "EfficientNetB0":
        target_size = (225, 225)
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif model_name == "ResNet50":
        target_size = (224, 224)
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif model_name == "VGG16":
        target_size = (224, 224)
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif model_name == "HybridNet":
        target_size = (224, 224)
        def preprocess_input(x): return x / 255.0
        expected_channels = 1
    elif model_name == "CustomCNN":
        target_size = (128, 128)
        def preprocess_input(x): return x / 255.0
    else:
        target_size = (224, 224)
        def preprocess_input(x): return x / 255.0

    if expected_channels == 1:
        img = img.convert("L")
        img = img.resize(target_size)
        img_array = np.array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=-1)
    else:
        img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img).astype("float32")

    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_emotion(model, meta, image_array):
    preds = model.predict(image_array)
    idx = np.argmax(preds[0])
    emotion = meta["emotion_labels"][idx]
    confidence = preds[0][idx] * 100
    return emotion, confidence


# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("🧭 Navigation Panel")
st.sidebar.markdown("""
Choose a page from below to explore different features of this AI-powered application.
""")

page = st.sidebar.radio("Go to", ["🏠 Home", "ℹ️ About", "😊 Detection", "📞 Contact"])


# ---------------------------
# HOME PAGE
# ---------------------------
if page == "🏠 Home":
    st.title("🎭 Emotion Detection Web App")

    st.markdown("""
    ### 👋 Welcome to the Future of Human–AI Interaction  
    This Emotion Detection App uses **cutting-edge deep learning models**  
    to analyze facial expressions and predict emotions with impressive accuracy.

    ### 🚀 Features You Can Explore  
    - **Upload and analyze any face image**  
    - **Live webcam detection** with real-time emotion predictions  
    - **Six powerful neural networks** running behind the scenes  
    - **Consensus voting system** for reliable results  
    - Supports **RGB** and **grayscale** deep learning models  

    ### 🧠 Models Integrated  
    ⭐ VGG16  
    ⭐ ResNet50  
    ⭐ InceptionV3  
    ⭐ EfficientNetB0  
    ⭐ HybridNet (custom grayscale-trained model)  
    ⭐ CustomCNN (trained from scratch)  

    ### 🎯 What Emotions Can Be Detected?
    - Happy  
    - Sad  
    - Angry  
    - Fear  
    - Disgust  
    - Surprise  
    - Neutral  

    Explore all functionalities from the sidebar.
    """)

    st.markdown("---")
    st.markdown("### 💡 *Tip:* For best webcam results, ensure good lighting!")


# ---------------------------
# ABOUT PAGE
# ---------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 🎯 Project Overview  
    This project demonstrates the real-world application of  
    **deep learning, computer vision, and real-time inference**.

    #### 🏗️ Built With:
    - **TensorFlow / Keras** → Deep Learning  
    - **OpenCV** → Face detection & live video processing  
    - **Streamlit** → Fast, responsive UI for ML apps  

    ## 🧠 Models Used in This Application
    ### 🔹 1. VGG16  
    A deep but simple architecture, excellent for transfer learning.

    ### 🔹 2. ResNet50  
    Uses skip connections to solve vanishing gradient problems.

    ### 🔹 3. InceptionV3  
    Captures features at multiple scales for higher accuracy.

    ### 🔹 4. EfficientNetB0  
    Lightweight, fast, and powerful — optimized CNN architecture.

    ### 🔹 5. HybridNet  
    A **custom-designed fusion model** trained on grayscale images  
    to enhance feature sensitivity.

    ### 🔹 6. CustomCNN  
    A fully custom convolutional neural network trained on FER-2013.

    ---

    ## 📚 Dataset Information  
    The models are trained on **FER-2013**, a widely used dataset containing:
    - 35,887 grayscale facial images  
    - 48×48 resolution  
    - 7 emotion labels  
    - Challenging variations in pose, lighting & expression  

    ---

    ## 🧩 Why This Project Matters  
    Emotion detection has applications in:
    - Healthcare  
    - Psychology  
    - Security and Surveillance  
    - Robotics  
    - Human-computer interaction  
    - Customer feedback analysis  

    This app combines all these ideas into one sleek, interactive platform.
    """)


# ---------------------------
# DETECTION PAGE
# ---------------------------
elif page == "😊 Detection":
    st.title("🧠 Emotion Detection Dashboard")
    st.subheader("Choose a detection mode below:")

    st.markdown("""
    ### 📥 1️⃣ Upload an Image  
    Upload any face photo (JPG/JPEG/PNG).  
    The system processes the image using all available models,  
    shows individual predictions, and also gives a **final consensus emotion**.

    ### 📸 2️⃣ Live Webcam Detection  
    Enable your camera to detect faces in real-time.  
    The HybridNet model is used for live predictions  
    along with bounding boxes and confidence percentages.

    ### ⚙️ *Internally, the app performs:*  
    - Face extraction  
    - Preprocessing using model-specific pipelines  
    - Deep learning inference  
    - Label decoding  
    - Confidence score computation  
    """)

    col1, col2 = st.columns(2)

    # -------- Upload Image Section --------
    with col1:
        st.markdown("## 🖼️ Upload Image")
        st.markdown("Upload a face image and let the AI identify the emotion.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write("🔍 Running predictions...")

            results = []
            for name in available_models:
                try:
                    channels = models[name].input_shape[-1]
                    img_array = preprocess_image(image, name, channels)
                    pred, conf = predict_emotion(models[name], metadata[name], img_array)
                    results.append((name, pred, conf))
                    st.success(f"**{name}** → {pred} {conf:.2f}%")
                except Exception as e:
                    st.error(f"⚠️ Error in {name}: {e}")

            if len(results) > 1:
                all_preds = [r[1] for r in results]
                majority = max(set(all_preds), key=all_preds.count)
                st.markdown(f"## 🧩 Final Consensus Result: **{majority}** ✔️")

    # -------- Live Detection Section --------
    with col2:
        st.markdown("## 🎥 Live Emotion Detection")
        st.markdown("Click the button below to start your webcam.")

        start_live = st.button("Start Webcam Detection")

        if start_live:
            st.info("To stop the camera, click 'Stop' in the Streamlit toolbar.")
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                                 + "haarcascade_frontalface_default.xml")

            model_name = "HybridNet"
            if model_name not in models:
                st.error("❌ HybridNet not loaded. Cannot start webcam detection.")
            else:
                expected_channels = models[model_name].input_shape[-1]

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Webcam not detected.")
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        face_array = preprocess_image(face_pil, model_name, expected_channels)

                        pred, conf = predict_emotion(models[model_name],
                                                     metadata[model_name],
                                                     face_array)

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{pred} ({conf:.1f}%)", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                cap.release()


# ---------------------------
# CONTACT PAGE
# ---------------------------
elif page == "📞 Contact":
    st.title("📞 Contact Developer")
    st.markdown("""
    Thank you for exploring this Emotion Detection App!  
    Feel free to reach out for collaborations, improvements, or questions.

    ## 👨‍💻 Developer Information
    **Your Name**  
    📧 Email: yourname@email.com  
    🌐 GitHub: github.com/yourusername  
    💼 LinkedIn: linkedin.com/in/yourprofile  

    ---

    ## 📬 Additional Information  
    - Open to research collaborations  
    - Available for ML/AI project assistance  
    - Currently expanding work in Computer Vision  
    - Interested in real-time AI and deep learning applications  

    ## 💬 Feedback & Suggestions  
    Your feedback helps improve this project.  
    Don’t hesitate to share your thoughts!

    ## ⭐ Future Enhancements Planned  
    - Multi-face emotion tracking  
    - Emotion analytics & charts  
    - Improved webcam FPS  
    - Mobile version of the app  
    - Transform-based modern models (ViT, SWIN)  

    Let’s build something amazing together! 🚀
    """)

