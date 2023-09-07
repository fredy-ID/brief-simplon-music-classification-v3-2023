import os
from django.conf import settings
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from .serializers import PredictSerializer, UserFeedbackSerializer
from .forms import UploadFileForm
from .models import Predict
import soundfile as sf


encoder = LabelEncoder()
scaler = StandardScaler()

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
        try:
            music = request.FILES['music']
        except Exception as e:
            return Response(
                {
                    'msg': "La musique n'a pas été acceptée",
                    'error': str(e),
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        response = super().post(request, *args, **kwargs)
        instance = response.data.get('id', None)
        
        # Découpage du fichier audio en plusieurs segemnts
        # music_file = open(settings.MEDIA_ROOT + "\\" +music.name)
        # y, sr = librosa.load(music_file)

        # Découpage en segments de 10 secondes
        # segment_duration = 10  # en secondes
        # segments = []
        # for start_time in range(0, len(y), int(segment_duration * sr)):
        #     end_time = min(start_time + int(segment_duration * sr), len(y))
        #     segment = y[start_time:end_time]
        #     segments.append(segment)
        
        
        
        # # Extraction de 30 secondes dans le fichier audio
        # desired_duration = 30
        # desired_start = 15
        # start_time = int(desired_start * sr)
        # end_time = int(desired_start+desired_duration * sr)
        # segment = y[start_time:end_time]
        
        # print("_________________________________")
        # print("segment", segment)
        # print("sr", sr)
        # print("start_time", start_time)
        # print("end_time", end_time)
        # print("_________________________________")
        
        return Response(
            {
                'msg': "Prédiction faite",
                'predicted_classes': "predicted_class_names",
                'id': instance
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
    
class UserFeedbackView(generics.CreateAPIView):
    serializer_class = UserFeedbackSerializer
    permission_classes = (AllowAny,)
    
    def post(self, request, *args, **kwargs):
        try:
            predict = Predict.objects.get(id=kwargs.get('id_predict'))
        except Predict.DoesNotExist:
            return Response({'error': "Cette prédiction n'existe pas"}, status=status.HTTP_404_NOT_FOUND)
        
        request.data['predict'] = predict.id
        self.create(request, *args, **kwargs)
    
        return Response(
            {
                'msg': "Merci pour votre feedback"
            },
            status=status.HTTP_200_OK
        )