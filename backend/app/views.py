import os
import csv
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from .serializers import PredictSerializer, UserFeedbackSerializer, RetrainingSerializer, FeatureSerializer, CSVDatasetSerializer
import joblib
from .forms import UploadFileForm
from .models import Predict, Features, CSVDataset, UserFeedback
import soundfile as sf
from django.db.models import Count
from sklearn.model_selection import train_test_split


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

        predicted = Predict.objects.create(feature = savedFeature, prediction = predicted_class_names[0])

        predictedSerialized = PredictSerializer(predicted)
        return Response(            {
                'msg': "Prédiction faite",
                'predicted_classes': predicted_class_names,
                'id': predictedSerialized.data.get("id")
            },
            status=status.HTTP_200_OK)

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
    
        return Response({'msg': "Merci pour votre feedback"},
            status=status.HTTP_200_OK
        )


class Dataset(generics.CreateAPIView):
    def get(self, request, *args, **kwargs):
        predictions = Predict.objects.values('prediction').annotate(dcount=Count('prediction'))
        serializedPrediction = {}

        for prediction in predictions:
            serializedPrediction[prediction['prediction']] = prediction["dcount"]

        
        return Response(serializedPrediction)
        
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
            # genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            genres = ["['blues']", "['classical']", "['country']", "['disco']", "['hiphop']", "['jazz']", "['metal']", "['pop']", "['reggae']", "['rock']"]
            predictions_by_genre = {}

            for genre in genres:
                predictions = Predict.objects.filter(csv=None, prediction=genre)
                for prediction in predictions:
                    predictions_by_genre[genre] = prediction.feature
                
                # return Response(
                #     {
                #         'msg': "test",
                #         "d": predictions_by_genre
                #     },
                #     status=status.HTTP_400_BAD_REQUEST
                # )
                
                # TODO: Ajout de la data récupéré dans le CSV
                new_features_df = pd.DataFrame({'predictions': predictions_by_genre[genre]})
                
                new_features_df['genre'] = genre

                csv_file_path = "app/CSVs/dataset.csv"
                if os.path.getsize(csv_file_path) == 0:
                    field_names = [
                        'chroma_stft_mean', 'chroma_stft_var',
                        'rms_mean', 'rms_var',
                        'spectral_centroids_mean', 'spectral_centroids_var',
                        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
                        'rolloff_mean', 'rolloff_var',
                        'zcr_mean', 'zcr_var',
                        'harmony_mean', 'harmony_var',
                        'tempo_mean', 'tempo_var',
                        'genre'  # Add the "genre" field
                    ]
                    with open(csv_file_path, 'w', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=field_names)
                        writer.writeheader()

                # csv_file_path = os.path.join(settings.BASE_DIR, 'CSVs', 'dataset.csv')
                existing_df = pd.read_csv(csv_file_path)
                final_df = pd.concat([existing_df, new_features_df], axis=1)
                
                final_df.to_csv(csv_file_path, index=False)
                
        except Exception as e:
            return Response(
                {
                    'msg': f"Impossible de récupérer les données ciblées : {str(e)}"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            x = np.array(final_df, dtype = float)
            x = scaler.fit_transform(final_df)
            y = encoder.fit_transform(genres)
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
            x_train.shape, x_test.shape, y_train.shape, y_test.shape
        except Exception as e:
            return Response(
                {
                    'msg': f"une erreur est survenue lors du train_test split: {str(e)}"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # TODO: Entraînement du modèle
    
            model = keras.models.Sequential([
                keras.layers.Dense(512, activation="relu", input_shape=(x_train.shape[1],)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256,activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(128,activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64,activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10, activation="softmax"),
            ])
            model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
            _, accuracy = model.evaluate(x_test, y_test, batch_size=128)
            epochs = 50
            
            try:
                history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=128)
            except:
                return Response(
                    {
                        'msg': "L'entraînement n'a pas été effectué"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            model.save("ai_models/test/stable_model/")
            model.save_weights('ai_models/test/stable_model/')
            
        except:
            return Response(
                {
                    'msg': "Une erreur est survenue lors de l'entraînement"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        self.create(request, *args, **kwargs)
    
        return Response(
            {
                'msg': "L'entraînement a été effectué",
                'epochs': epochs,
                'accuracy': accuracy,
                'history': history
            },
            status=status.HTTP_200_OK
        )
    
