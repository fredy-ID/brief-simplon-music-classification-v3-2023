import os
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from .serializers import PredictSerializer, UserFeedbackSerializer, RetrainingSerializer, FeatureSerializer, CSVDatasetSerializer
import joblib
from .forms import UploadFileForm
from .models import Predict, Features, CSVDataset, Retraining
import soundfile as sf


encoder = LabelEncoder()
scaler = joblib.load("./app/ai_models/scalerModelF.pkl")

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
num_classes = len(class_names)


def get_3sec_sample(uploaded_audio):
    audio, sample_rate = librosa.load(
        uploaded_audio,
        sr=None,
    )

    segment_duration = 3  # Durée de chaque segment en secondes
    segment_length = int(sample_rate * segment_duration)
    segments = []

    # Découpage
    for i in range(0, len(audio), segment_length):
        segment = audio[i : i + segment_length]
        segments.append(segment)

    return segments


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
        # audio, sr = librosa.load(music, sr=None)

        audio = get_3sec_sample(music)  # Récupère un tableau de features pour chaque 3 seconde de la musique
        features = audio_pipeline(audio[2])  # Extrait les caractéristiques audio


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
        return Response(            {
                'msg': "Prédiction faite",
                'predicted_classes': predicted_class_names,
                'id': predictedSerialized.data.get("id")
            },
            status=status.HTTP_200_OK)

    
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
        
class CreateCsvView(generics.CreateAPIView):
    serializer_class = UserFeedbackSerializer
    permission_classes = (AllowAny,)
    
    def post(self, request, *args, **kwargs):
        
        # TODO: Création du CSV en physique
        # TODO: Ajout d'un pourcentage de data par genre dans le CSV
    
        self.create(request, *args, **kwargs)

        return Response(
            {
                'msg': "CSV créé"
            },
            status=status.HTTP_200_OK
        )
        
    
class RetrainingView(generics.CreateAPIView):
    serializer_class = RetrainingSerializer
    permission_classes = (AllowAny,)
    
    def post(self, request, *args, **kwargs):
        
            # TODO: Récupération de la data par nombre définit par l'utilisateur
        try:
            # Récupération de la data par nombre défini par l'utilisateur
            genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            predictions_by_genre = {}

            for genre in genres:
                predictions = Predict.objects.filter(csv=None, feature__genre=genre)[:request.POST['num_data_to_csv']]
                predictions_by_genre[genre] = [prediction.prediction for prediction in predictions]
                
        except Exception as e:
            return Response(
                {
                    'msg': f"Impossible de récupérer les données ciblées : {str(e)}"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # TODO: Ajout de la data récupéré dans le CSV
        new_features_df = pd.DataFrame({'predictions': predictions_by_genre[genre]})

        existing_df = pd.read_csv("./CSVs/dataset.csv")
        final_df = pd.concat([existing_df, new_features_df], axis=1)
        
        try:
            # TODO: Entraînement du modèle
            pass
        except:
            return Response(
                {
                    'msg': "L'entraînement n'a pas été effectué"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        self.create(request, *args, **kwargs)
    
        return Response(
            {
                'msg': "L'entraînement a bien été effectué"
            },
            status=status.HTTP_200_OK
        )
    