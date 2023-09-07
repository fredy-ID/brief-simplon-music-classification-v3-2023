import os
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from .serializers import PredictSerializer, FeatureSerializer
import joblib
from .models import Predict, Features

encoder = LabelEncoder()
scaler = joblib.load("./app/ai_models/scalerModelF.pkl")

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
num_classes = len(class_names)

try:
    model = load_model('app/ai_models/stable_model')
except Exception as e:
    print('_____________________________')
    print('Erreur lors du chargement du modèle :', e)
    print('_____________________________')
    raise

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

    tempo = librosa.beat.tempo(y=audio)
    features.append(tempo[0])

    # Calcul des moyennes des MFCC
    mfcc = librosa.feature.mfcc(y=audio)

    for x in mfcc:
        features.append(np.mean(x))
        features.append(np.var(x))

    return features

class PredictView(generics.CreateAPIView):
    serializer_class = PredictSerializer
    permission_classes = (AllowAny,)
    
    
    def post(self, request, *args, **kwargs):
        
        music = request.FILES['music']  # Suppose que la musique est téléchargée en tant que fichier
        print(music)
        # return Response("music")
        audio, sr = librosa.load(music, sr=None)  # Charge le fichier audio
        features = audio_pipeline(audio)  # Extrait les caractéristiques audio

        print("type", type(features[0]))
        data = {
            "chroma_stft_mean": features[0],
            "chroma_stft_var": features[1],

            "rms_mean": features[2],
            "rms_var": features[3],

            "spectral_centroids_mean": features[4],
            "spectral_centroids_var": features[5],

            "spectral_bandwidth_mean": features[6],
            "spectral_bandwidth_var": features[7],

            "rolloff_mean": features[8],
            "rolloff_var": features[9],

            "zcr_mean": features[10],
            "zcr_var": features[11],

            "harmony_mean": features[12],
            "harmony_var": features[13],

            "tempo_mean": features[14],
            "tempo_var": features[15],
        }
        savedFeature = Features.objects.create(
            chroma_stft_mean = features[0],
            chroma_stft_var = features[1],
            rms_mean = features[2],
            rms_var = features[3],
            spectral_centroids_mean = features[4],
            spectral_centroids_var = features[5],
            spectral_bandwidth_mean = features[6],
            spectral_bandwidth_var = features[7],
            rolloff_mean = features[8],
            rolloff_var = features[9],
            zcr_mean = features[10],
            zcr_var = features[11],
            harmony_mean = features[12],
            harmony_var = features[13],
            tempo_mean = features[14],
            tempo_var = features[15],
        
        )

        serialized_predict = FeatureSerializer(savedFeature)

        x_t = np.array(features, dtype=float).reshape(1, -1)  # Transforme les caractéristiques en tableau 2D
        # return Response(x_t)
        # Échelonne les données
        x_t = scaler.transform(x_t)

        # Prédiction
        prediction = model.predict(x_t)

        probs = np.exp(prediction) / np.sum(np.exp(prediction), axis=1, keepdims=True)
        predicted_classes = np.argmax(probs, axis=1)
        predicted_class_names = [class_names[class_index] for class_index in predicted_classes]
        print(f"Prédiction pour la musique : {predicted_class_names}")

        # instance = self.create(request, *args, **kwargs)

        predicted = Predict.objects.create(feature = savedFeature, prediction = predicted_class_names)

        predictedSerialized = PredictSerializer(predicted)
        return Response(predictedSerialized.data)

        return Response(
            {
                'msg': "Prédiction faite",
                'predicted_classes': predicted_class_names,
                # 'id': instance.id
            },
            status=status.HTTP_200_OK
        )
    
    # def post(self, request, *args, **kwargs):
    #     try:
    #         music = request.FILES['music']  # Suppose que la musique est téléchargée en tant que fichier
    #         audio, sr = librosa.load(music, sr=None)  # Charge le fichier audio
    #         features = audio_pipeline(audio)  # Extrait les caractéristiques audio
    #         x_t = np.array(features, dtype=float).reshape(1, -1)  # Transforme les caractéristiques en tableau 2D

    #         # Échelonne les données
    #         x_t = scaler.transform(x_t)

    #         # Prédiction
    #         prediction = model.predict(x_t)

    #         probs = np.exp(prediction) / np.sum(np.exp(prediction), axis=1, keepdims=True)

    #         predicted_classes = np.argmax(probs, axis=1)
    #         predicted_class_names = [class_names[class_index] for class_index in predicted_classes]

    #         print(f"Prédiction pour la musique : {predicted_class_names}")
            
    #         instance = self.create(request, *args, **kwargs)

    #         return Response(
    #             {
    #                 'msg': "Prédiction faite",
    #                 'predicted_classes': predicted_class_names,
    #                 'id': instance.id
    #             },
    #             status=status.HTTP_200_OK
    #         )
    #     except Exception as e:
    #         print('Erreur lors de la prédiction :', e)
    #         return Response(
    #             {
    #                 'msg': "Erreur lors de la prédiction",
    #                 'error': str(e)
    #             },
    #             status=status.HTTP_400_BAD_REQUEST
    #         )