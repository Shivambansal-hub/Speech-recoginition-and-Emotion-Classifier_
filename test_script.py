import numpy as np
import librosa
import tensorflow as tf
import joblib
from keras.models import load_model

# Constants
MAX_LEN = 300
FEATURE_DIM = 45
SAMPLE_RATE = 16000
CLASS_NAMES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Load trained model and scaler
model = load_model("model/best_model.keras")
scaler = joblib.load("scaler.pkl")

def load_and_preprocess(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    return y

def extract_features(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    try:
        f0 = librosa.yin(y=y, sr=SAMPLE_RATE, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')).reshape(1, -1)
    except:
        f0 = np.zeros((1, mfcc.shape[1]))

    T = mfcc.shape[1]
    def resize(feat): return feat[:, :T] if feat.shape[1] >= T else np.pad(feat, ((0, 0), (0, T - feat.shape[1])), mode='constant')

    all_features = np.vstack([
        mfcc, delta1, delta2,
        resize(centroid), resize(bandwidth), resize(rolloff),
        resize(rms), resize(zcr), resize(f0)
    ])
    features = all_features.T

    if features.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:MAX_LEN, :]

    return features

def predict_emotion(audio_path):
    y = load_and_preprocess(audio_path)
    X = extract_features(y)
    X_scaled = scaler.transform(X.reshape(-1, FEATURE_DIM)).reshape(1, MAX_LEN, FEATURE_DIM)

    prediction = model.predict(X_scaled)
    predicted_class_index = np.argmax(prediction)
    predicted_emotion = CLASS_NAMES[predicted_class_index]

    return predicted_emotion, prediction[0]

# Example usage:
if __name__ == "__main__":
    audio_file = "test.wav"  # Replace with your test file
    emotion, probs = predict_emotion(audio_file)
    print("Predicted Emotion:", emotion)
    print("Class Probabilities:", probs)
