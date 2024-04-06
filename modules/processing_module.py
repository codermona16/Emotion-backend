import os
import subprocess
import numpy as np
import librosa
from keras.models import load_model
def processAudio(path):
    try:
        saved_model_path = 'saved_models/audio_classification.keras'
        loaded_model = load_model(saved_model_path)

        def extract_features(audio_file_path):
        # Load audio file
            audio_data, sample_rate = librosa.load(audio_file_path, res_type='kaiser_fast') 
            # Extract features
            mfccs_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            return mfccs_scaled_features.reshape(1, -1)  # Reshape features to match model input shape

        # Example audio file path
        audio_file_path = path
        # Preprocess the audio data
        input_data = extract_features(audio_file_path)

        # Perform predictions using the loaded model
        predictions = loaded_model.predict(input_data)

        # Interpret the predictions
        # Assuming the model predicts probabilities for each class, you can print the predicted class
        predicted_class_index = np.argmax(predictions)
        class_labels = ['Happy','Sad','Angry','Surprise','Neutral']  # Define your class labels
        predicted_class = class_labels[predicted_class_index]
        return "Predicted Class : " + predicted_class
    except Exception as e:
        return "Error processing video: " +e

def process(filename):
    try:
        path = 'static/' + filename
        return processAudio(path)
    except Exception as e:
        return "Error in processing: " +e
