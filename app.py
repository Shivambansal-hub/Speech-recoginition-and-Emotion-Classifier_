import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os

# Constants
SAMPLE_RATE = 16000
MAX_LEN = 300
FEATURE_DIM = 45

# Load model and supporting files
model = load_model("model/best_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")  # ✅ Load encoder
class_names = list(encoder.classes_)

# Helper Functions
def load_and_preprocess(file_path, sr=SAMPLE_RATE):
    y, _ = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    return y

def extract_features(y, sr=SAMPLE_RATE, max_len=MAX_LEN):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    try:
        f0 = librosa.yin(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = f0.reshape(1, -1)
    except:
        f0 = np.zeros((1, mfcc.shape[1]))

    T = mfcc.shape[1]
    def resize(feat): return feat[:, :T] if feat.shape[1] >= T else np.pad(feat, ((0, 0), (0, T - feat.shape[1])), mode='constant')
    
    features = np.vstack([mfcc, delta1, delta2, resize(centroid), resize(bandwidth),
                          resize(rolloff), resize(rms), resize(zcr), resize(f0)]).T

    # Pad or truncate
    if features.shape[0] < max_len:
        features = np.pad(features, ((0, max_len - features.shape[0]), (0, 0)), mode='constant')
    else:
        features = features[:max_len, :]

    return features

# Prediction function
def predict_emotion(file_path):
    y = load_and_preprocess(file_path)
    features = extract_features(y)
    features = scaler.transform(features)
    features = np.expand_dims(features, axis=0)  # shape: (1, 300, 45)
    probs = model.predict(features)[0]
    predicted_label = class_names[np.argmax(probs)]
    return predicted_label, probs

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition Web App")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav", format="audio/wav")

    if st.button("Predict Emotion"):
        predicted_emotion, probabilities = predict_emotion("temp.wav")
        st.success(f"🎯 Predicted Emotion: **{predicted_emotion}**")

        # Plot probabilities
        fig, ax = plt.subplots()
        ax.barh(class_names, probabilities, color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)
