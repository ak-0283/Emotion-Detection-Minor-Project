import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import cv2
from PIL import Image
import os

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
def load_all_models(model_dir="models"):
    """Load all available CNN models and handle both RGB and grayscale models."""
    model_names = ["VGG16", "ResNet50", "InceptionV3", "EfficientNetB0"]
    models, metadata, available = {}, {}, []

    st.subheader("📦 Model Loading Status")

    for name in model_names:
        model_path = os.path.join(model_dir, f"{name}_emotion_model.keras")
        meta_path = os.path.join(model_dir, f"{name}_metadata.pkl")

        st.write(f"🔍 **Checking:** `{model_path}`")

        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                input_shape = model.input_shape

                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)

                models[name] = model
                metadata[name] = meta
                available.append(name)

                # Display success in main area
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
            st.warning(f"⚠️ Missing files for **{name}**. Check your 'models' folder.")

    if not available:
        st.error("🚫 No valid models loaded — check your model folder.")
    else:
        st.success(f"🎉 Successfully loaded {len(available)} models: {', '.join(available)}")

    return models, metadata, available




# Button to reload models dynamically
if st.sidebar.button("🔄 Reload Models"):
    st.cache_resource.clear()
    st.experimental_rerun()

models, metadata, available_models = load_all_models()


# ---------------------------
# IMAGE PREPROCESSING (supports RGB + Grayscale)
# ---------------------------
def preprocess_image(img, model_name="VGG16", expected_channels=3):
    """Resize and preprocess image according to model input and expected channels."""
    if model_name == "InceptionV3":
        target_size = (299, 299)
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif model_name == "EfficientNetB0":
        target_size = (225, 225)
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif model_name == "ResNet50":
        target_size = (224, 224)
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        target_size = (224, 224)
        from tensorflow.keras.applications.vgg16 import preprocess_input

    # Handle grayscale or RGB
    if expected_channels == 1:
        img = img.convert("L")  # convert to grayscale
        img = img.resize(target_size)
        img_array = np.array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=-1)  # add channel dimension
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
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "ℹ️ About", "😊 Detection", "📞 Contact"])


# ---------------------------
# HOME PAGE
# ---------------------------
if page == "🏠 Home":
    st.title("🎭 Emotion Detection using Deep Learning")
    st.markdown("""
    Welcome to the **Emotion Detection App** — powered by **4 pretrained CNN models**:
    - 🧠 VGG16  
    - ⚡ ResNet50  
    - 🔍 InceptionV3  
    - 🌿 EfficientNetB0  

    Detect emotions such as **happy**, **sad**, **angry**, **neutral**, **fear**, **disgust**, and **surprise**  
    from static images or live webcam input.

    👉 Use the sidebar to navigate between pages.
    """)


# ---------------------------
# ABOUT PAGE
# ---------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    This web app uses **TensorFlow**, **Keras**, and **Streamlit** to detect facial emotions.

    ### 🧠 Models Used
    - **VGG16:** Classic CNN baseline (RGB)
    - **ResNet50:** Deep residual network (RGB)
    - **InceptionV3:** Multi-scale CNN (RGB)
    - **EfficientNetB0:** Compact, grayscale-trained CNN (1-channel)

    ### 🎯 Features
    - Upload an image for prediction  
    - Real-time emotion detection via webcam  
    - Ensemble voting across multiple models  

    Dataset: **FER-2013 (Facial Expression Recognition)**  
    Includes 7 emotions: happy, sad, angry, neutral, fear, disgust, surprise
    """)


# ---------------------------
# DETECTION PAGE
# ---------------------------
elif page == "😊 Detection":
    st.title("🧠 Emotion Detection Dashboard")
    st.subheader("Choose a detection mode:")

    col1, col2 = st.columns(2)

    # -------- Upload Image Section --------
    with col1:
        st.markdown("### 🖼️ Upload an Image")
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write("🔍 Detecting emotions using available models...")

            results = []
            for name in available_models:
                try:
                    channels = models[name].input_shape[-1]
                    img_array = preprocess_image(image, name, channels)
                    pred, conf = predict_emotion(models[name], metadata[name], img_array)
                    results.append((name, pred, conf))
                    st.success(f"**{name}** → {pred} ({conf:.2f}%)")
                except Exception as e:
                    st.error(f"⚠️ {name} failed: {e}")

            if len(results) > 1:
                all_preds = [r[1] for r in results]
                majority = max(set(all_preds), key=all_preds.count)
                st.markdown(f"### 🧩 Final Consensus: **{majority}**")

    # -------- Live Detection Section --------
    with col2:
        st.markdown("### 📷 Live Webcam Detection")
        start_live = st.button("Start Webcam Detection")

        if start_live:
            st.info("Press 'Stop' in Streamlit toolbar to stop webcam feed.")
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            model_name = "EfficientNetB0"  # Default webcam model
            if model_name not in models:
                st.error("❌ EfficientNetB0 not loaded. Cannot start webcam detection.")
            else:
                expected_channels = models[model_name].input_shape[-1]

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("❌ Cannot access webcam.")
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        face_array = preprocess_image(face_pil, model_name, expected_channels)
                        pred, conf = predict_emotion(models[model_name], metadata[model_name], face_array)

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
    ### 👨‍💻 Developer Info
    **Your Name**  
    📧 Email: [yourname@email.com](mailto:yourname@email.com)  
    🌐 GitHub: [github.com/yourusername](https://github.com/yourusername)  
    💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

    💬 Have feedback or collaboration ideas? Feel free to reach out!
    """)
