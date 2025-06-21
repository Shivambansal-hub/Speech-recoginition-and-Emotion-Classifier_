# 🎙️ Speech Emotion Recognition Web App 😄😢😠

This project implements a **deep learning-based web application** that detects **human emotions from speech** using a trained **CNN model**. The app allows users to upload `.wav` audio files, processes the speech, and classifies it into one of **eight emotions**.

---

## 🚀 Demo

> 📽️ A short 2-minute demo video showcasing the web app’s functionality will be provided.

---

## 🧠 Emotion Classes

The model classifies speech into the following 8 emotions:

- `Neutral`
- `Calm`
- `Happy`
- `Sad`
- `Angry`
- `Fearful`
- `Disgust`
- `Surprised`

---

## 📂 Project Structure

```
Speech_Emotion_Recognition/
├── app.py                  # Streamlit web app
├── model/best_model.keras          # Trained CNN model
├── scaler.pkl              # Scaler used for feature normalization
├── test_model.py
|-- encoder.pkl         # Script to test the model with any .wav file
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── demo_video.mp4          # (Optional) Demo video
```

---

## 📊 Dataset
I have included Dataset in this repository

---

## 🧪 Preprocessing Methodology

1. **Audio Loading**:  
   Audio files were loaded using `librosa` with a sample rate of **16 kHz** and trimmed for silence.

2. **Feature Extraction**:
   Each audio file was converted to a feature vector of shape **(300, 45)** using:
   - **MFCCs (13)**
   - **Delta MFCCs (13)**
   - **Delta-Delta MFCCs (13)**
   - **Spectral features**: Centroid, Bandwidth, Rolloff, RMS, ZCR (each 1)
   - **Pitch (YIN)**

3. **Padding**:  
   Features were padded or truncated to ensure fixed length (300 timesteps).

4. **Normalization**:  
   `StandardScaler` was applied across all time-step vectors.

5. **Label Encoding**:  
   String labels (like 'happy', 'sad') were encoded to integers and one-hot encoded.

6. **Handling Class Imbalance**:  
   **SMOTE** was used to oversample minority classes.

---

## 🧠 Model Architecture (CNN Based)

```
Input: (300, 45)
↓
Conv1D (128 filters, kernel_size=5) + BatchNorm + MaxPooling + Dropout(0.4)
↓
Conv1D (256 filters, kernel_size=5) + BatchNorm + MaxPooling + Dropout(0.4)
↓
Flatten → Dense(128) + Dropout(0.5)
↓
Output Dense Layer (Softmax) → 8 Classes
```

- **Loss Function**: Categorical Crossentropy with label smoothing
- **Optimizer**: Adam
- **Regularization**: L2 and dropout
- **Epochs**: 100
- **Validation Split**: 15%

---

## ✅ Model Performance

| Metric            | Value   |
|-------------------|---------|
| Training Accuracy | ~97%    |
| Validation Acc.   | ~88%    |
| Evaluation        | Confusion Matrix, Accuracy, F1-Score |

📊 A confusion matrix visualization is also included in the training notebook.

---

## 🌐 Web App Features

- Upload `.wav` files directly
- View predicted emotion
- View probability distribution across all 8 emotions
- Audio preview of uploaded file
- Lightweight and responsive interface

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Shivambansal-hub/Speech-recoginition-and-Emotion-Classifier_
cd Speech-recoginition-and-Emotion-Classifier_
```

### 2. Create & Activate Environment
```bash
python -m venv emotion_env
source emotion_env/bin/activate   # Mac/Linux
emotion_env\Scripts\activate      # Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App
```bash
python -m streamlit run app.py
```

---

## 🎯 Test the Model on Audio Files

Use the `test_model.py` script:
```bash
python test_model.py --file sample.wav
```

This will display the predicted emotion and class probabilities.

---

## 💡 Future Work

- Support microphone input recording directly
- Real-time emotion streaming
- Use larger datasets for better generalization
- Try transformer-based models

---

## 📧 Contact

Developed by **Shivam Bansal**  
Feel free to connect or report issues at: shivambansal357@gmail.com
