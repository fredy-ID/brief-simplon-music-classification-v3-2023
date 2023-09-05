import os
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
encoder = LabelEncoder()
scaler = StandardScaler()

from .serializers import PredictSerializer

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop','jazz', 'metal', 'pop', 'reggae', 'rock']
num_classes = len(class_names)

try:
    model = load_model('app/ai_models/modelF.hdf5')
except Exception as e:
    print('_____________________________')
    print('Erreur lors du chargement du modèle :', e)
    print('_____________________________')
    raise

class PredictView(generics.CreateAPIView):
    serializer_class = PredictSerializer
    permission_classes = (AllowAny,)


    def create(self, request, *args, **kwargs):
        
        df = audio_pipeline(music)
        
        df_labels = df.iloc[:, -1] # get labels column
        df_predict = df.iloc[:, :-1]
        
         # assign x and y, scale x and encode y
        x_t = np.array(df_predict, dtype = float)
        x_t = scaler.fit_transform(df_predict)
        y_t = encoder.fit_transform(df_labels)
        joblib.dump(scaler, "scaler1.pkl")

        x_t.shape, y_t.shape
        
        prediction = model.predict(x_t)

        probs = np.exp(prediction) / np.sum(np.exp(prediction), axis=1, keepdims=True)

        predicted_classes = np.argmax(probs, axis=1)
        predicted_class_names = [class_names[class_index] for class_index in predicted_classes]

        print(f"Prédiction pour la musique : {predicted_class_names}")
        
        return Response(
            {
                'msg': "prédiction faite"
            }, 
            status=status.HTTP_200_OK
        )
    
    def audio_pipeline(audio):

        features = []

        # Calcul du ZCR
        
        chroma_stft = librosa.feature.chroma_stft(y=audio)
        features.append(np.mean(chroma_stft))
        features.append(np.var(chroma_stft))
        
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.var(rms))
        
        # Calcul de la moyenne du Spectral centroid
        
        # spectral_centroids = librosa.feature.spectral_centroid(y=audio)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio)
        features.append(np.mean(spectral_centroids))
        features.append(np.var(spectral_centroids))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio)
        features.append(np.mean(spectral_bandwidth))
        features.append(np.var(spectral_bandwidth))

        rolloff = librosa.feature.spectral_rolloff(y=audio)
        features.append(np.mean(rolloff))
        features.append(np.var(rolloff))

        zcr = librosa.feature.zero_crossing_rate(y=audio)
        features.append(np.mean(zcr))
        features.append(np.var(zcr))
        
        harmony = librosa.effects.harmonic(y=audio)
        features.append(np.mean(harmony))
        features.append(np.var(harmony))
        
        tempo = librosa.feature.tempo(y=audio)
        features.append(tempo[0])

        # Calcul des moyennes des MFCC

        mfcc = librosa.feature.mfcc(y=audio)

        for x in mfcc:
            features.append(np.mean(x))
            features.append(np.var(x))


        return features