import gradio as gr
import numpy as np
import cv2
import librosa
import moviepy.editor as mp
import speech_recognition as sr
import tempfile
import wave
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model, model_from_json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from collections import Counter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# Load the text model
with open('model_architecture_for_text_emotion_updated_json.json', 'r') as json_file:
    model_json = json_file.read()
text_model = model_from_json(model_json)
text_model.load_weights("model_for_text_emotion_updated(1).keras")

# Load the encoder and scaler for audio
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the tokenizer for text
with open('tokenizer.json') as json_file:
    tokenizer_json = json.load(json_file)
tokenizer = tokenizer_from_json(tokenizer_json)

# Load the audio model
audio_model = load_model('my_model.h5')

# Load the image model
image_model = load_model('model_emotion.h5')

# Initialize NLTK
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess text function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

# Extract features from audio
def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

# Predict emotion from text
def find_emotion_using_text(sample_rate, audio_data, recognizer):
    mapping = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        
        with wave.open(temp_audio_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
    with sr.AudioFile(temp_audio_path) as source:
        audio_record = recognizer.record(source)
        text = recognizer.recognize_google(audio_record)
        pre_text = preprocess_text(text)
        title_seq = tokenizer.texts_to_sequences([pre_text])
        padded_title_seq = pad_sequences(title_seq, maxlen=35, padding='post', truncating='post')
        inp1 = np.array(padded_title_seq)
        text_prediction = text_model.predict(inp1)
    
    os.remove(temp_audio_path)
    max_index = text_prediction.argmax()
    return mapping[max_index]

# Predict emotion from audio
def predict_emotion(audio_data):
    sample_rate, data = audio_data
    data = data.flatten()
    
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    data = data / np.max(np.abs(data))
    
    features = extract_features(data, sample_rate)
    features = np.expand_dims(features, axis=0)
    
    if features.ndim == 3:
        features = np.squeeze(features, axis=2)
    elif features.ndim != 2:
        raise ValueError("Features array has unexpected dimensions.")
    
    scaled_features = scaler.transform(features)
    scaled_features = np.expand_dims(scaled_features, axis=2)
    
    prediction = audio_model.predict(scaled_features)
    emotion_index = np.argmax(prediction)
    
    num_classes = len(encoder.categories_[0])
    emotion_array = np.zeros((1, num_classes))
    emotion_array[0, emotion_index] = 1
    
    emotion_label = encoder.inverse_transform(emotion_array)[0]
    return emotion_label

# Preprocess image
def preprocess_image(image):
    image = load_img(image, target_size=(48, 48), color_mode="grayscale")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Predict emotion from image
def predict_emotion_from_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = image_model.predict(preprocessed_image)
    emotion_index = np.argmax(prediction)
    
    mapping = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
    return mapping[emotion_index]

# Main function to handle text, audio, and image emotion recognition
# Load the models and other necessary files (as before)

# Preprocess image (as before)

# Predict emotion from image (as before)

# Extract features from audio (as before)

# Predict emotion from text (as before)

# Predict emotion from audio (as before)

# Extract frames from video and predict emotions
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    predictions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process every nth frame (to speed up processing)
        if frame_count % int(frame_rate) == 0:
            # Convert frame to grayscale as required by your model
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (48, 48))  # Resize to match model input size
            frame = img_to_array(frame)
            frame = np.expand_dims(frame, axis=0) / 255.0
            
            # Predict emotion
            prediction = image_model.predict(frame)
            predictions.append(np.argmax(prediction))
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Find the most common prediction
    most_common_emotion = Counter(predictions).most_common(1)[0][0]
    mapping = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
    return mapping[most_common_emotion]

# Process audio from video and predict emotions
def process_audio_from_video(video_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        audio.write_audiofile(temp_audio_path)
        
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio_record = recognizer.record(source)
        text = recognizer.recognize_google(audio_record)
        pre_text = preprocess_text(text)
        title_seq = tokenizer.texts_to_sequences([pre_text])
        padded_title_seq = pad_sequences(title_seq, maxlen=35, padding='post', truncating='post')
        inp1 = np.array(padded_title_seq)
        text_prediction = text_model.predict(inp1)
    
    os.remove(temp_audio_path)
    
    max_index = text_prediction.argmax()
    text_emotion = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}[max_index]
    
    audio_emotion = predict_emotion((audio.fps, np.array(audio.to_soundarray())))
    
    return text_emotion, audio_emotion

# Main function to handle video emotion recognition
def transcribe_and_predict_video(video):
    # Process video frames for image-based emotion recognition
    image_emotion = process_video(video)
    
    # Process audio for text and audio-based emotion recognition
    text_emotion, audio_emotion = process_audio_from_video(video)
    
    return f"Text Emotion: {text_emotion}, Audio Emotion: {audio_emotion}, Image Emotion: {image_emotion}"




gui = gr.Interface(
    fn=transcribe_and_predict_video,
    inputs=gr.File(label="Upload Video"),
    outputs=gr.Textbox(label="Detected Emotion"),
    title="AI Video Emotion Analyzer"
)

gui.launch()



import uvicorn
from fastapi import FastAPI, File, UploadFile
import shutil
import os


# FastAPI Backend
app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "API is running"}
    
@app.post("/predict_emotion/")
async def predict_emotion_api(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(file.file.read())
        video_path = temp_file.name
    
    emotions=transcribe_and_predict_video(video_path)
    os.remove(video_path)
    
    return emotions
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=PORT)




