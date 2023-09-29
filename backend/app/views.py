from .serializers import PredictSerializer, UserFeedbackSerializer, RetrainingSerializer
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from rest_framework.permissions import AllowAny
from sklearn.preprocessing import LabelEncoder
from rest_framework.response import Response
from rest_framework import generics, status
from tensorflow.keras.models import load_model
from .models import Predict, Features
from django.db.models import Count
from tensorflow import keras
import soundfile as sf
import librosa.display
import pandas as pd
import numpy as np
import librosa
import joblib
import csv
import os


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
        
    


# Limit représente le nombre limite de prediction par genre nécessaire pour l'extraction vers le CSV
def extractFeatureFromDBToCSV(limit = 2):
    print('in extract features  ')
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    feature_of_predictions_by_genre = {}
    ids_to_update = []
    for genre in genres:
        predictions = Predict.objects.filter(exists_in_csv=False, prediction=genre)[:limit]
        
        # Pourquoi un array ici, car pour X genre on peux avoir plusieur feature à extraire
        # Le nombre de feature à extraire étant définit par limit
        feature_of_predictions_by_genre[genre] = []

        for prediction in predictions:
            feature_of_predictions_by_genre[genre].append({    
                "chroma_stft_mean": prediction.feature.chroma_stft_mean,
                "chroma_stft_var": prediction.feature.chroma_stft_var,
                "rms_mean":        prediction.feature.rms_mean,
                "rms_var" :        prediction.feature.rms_var,
                "spectral_centroids_mean": prediction.feature.spectral_centroids_mean,
                "spectral_centroids_var": prediction.feature.spectral_centroids_var,
                "spectral_bandwidth_mean": prediction.feature.spectral_bandwidth_mean,
                "spectral_bandwidth_var": prediction.feature.spectral_bandwidth_var,
                "rolloff_mean" : prediction.feature.rolloff_mean,
                "rolloff_var": prediction.feature.rolloff_var,
                "zcr_mean": prediction.feature.zcr_mean,
                "zcr_var": prediction.feature.zcr_var,
                "harmony_mean": prediction.feature.harmony_mean,
                "harmony_var": prediction.feature.harmony_var,
                "tempo_mean": prediction.feature.tempo_mean,
                "tempo_var": prediction.feature.tempo_var 
            })
            ids_to_update.append(prediction.id)
        
    data = {
        "chroma_stft_mean": [],
        "chroma_stft_var": [],
        "rms_mean": [],
        "rms_var": [],
        "spectral_centroids_mean": [],
        "spectral_centroids_var": [],
        "spectral_bandwidth_mean": [],
        "spectral_bandwidth_var": [],
        "rolloff_mean": [],
        "rolloff_var": [],
        "zcr_mean": [],
        "zcr_var": [],
        "harmony_mean": [],
        "harmony_var": [],
        "tempo_mean": [],
        "tempo_var": [],
        "genre": [],
    }

    df = pd.DataFrame(data)

    for genderFeatures in feature_of_predictions_by_genre:
        for feature in feature_of_predictions_by_genre[genderFeatures]:
            # print(feature)
            feature['genre'] = genderFeatures
            series = pd.Series(feature)
            df = pd.concat([df, series.to_frame().T])

    return df, ids_to_update

class RetrainingView(generics.CreateAPIView):
    serializer_class = RetrainingSerializer
    permission_classes = (AllowAny,)
    
    def post(self, request, *args, **kwargs):
        
        # TODO: Récupération de la data par nombre définit par l'utilisateur
        dataframe, ids_to_update = extractFeatureFromDBToCSV()
        try:

            # TODO: Ajout de la data récupéré dans le CSV
            csv_file_path = "app/CSVs/dataset.csv"

            try:
                existing_df = pd.read_csv(csv_file_path)
            except FileNotFoundError:
                header = [
                    "chroma_stft_mean",
                    "chroma_stft_var",
                    "rms_mean",
                    "rms_var",
                    "spectral_centroids_mean",
                    "spectral_centroids_var",
                    "spectral_bandwidth_mean",
                    "spectral_bandwidth_var",
                    "rolloff_mean",
                    "rolloff_var",
                    "zcr_mean",
                    "zcr_var",
                    "harmony_mean",
                    "harmony_var",
                    "tempo_mean",
                    "tempo_var",
                    "genre"
                ]
                existing_df = pd.DataFrame(columns=header)
            
            existing_df.reset_index(drop=True, inplace=True)
            dataframe.reset_index(drop=True, inplace=True)
            final_df = pd.concat([existing_df, dataframe], axis=0)
            final_df.to_csv(csv_file_path, index=False)
            
            Predict.objects.filter(id__in=ids_to_update).update(exists_in_csv=True)
                
        except Exception as e:
            print("____________________________________")
            print("Impossible de récupérer les données ciblées")
            print(e)
            print("____________________________________")
            return Response(
                {
                    'msg': f"Impossible de récupérer les données ciblées : {str(e)}"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            y = final_df['genre']
            x = final_df.drop(columns=['genre'])
            x = np.array(x, dtype = float)
            x = scaler.fit_transform(x)
            y = encoder.fit_transform(y)
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
            x_train.shape, x_test.shape, y_train.shape, y_test.shape
        except Exception as e:
            print("____________________________________")
            print("une erreur est survenue lors du train_test split")
            print(e)
            print("____________________________________")
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
                print("____________________________________")
                print("L'entraînement n'a pas été effectué")
                print(e)
                print("____________________________________")
                return Response(
                    {
                        'msg': "L'entraînement n'a pas été effectué"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            model.save("app/ai_models/test/stable_model/")
            model.save_weights('app/ai_models/test/stable_model/')
            
        except Exception as e:
            print("____________________________________")
            print("Une erreur est survenue lors de l'entraînement")
            print(e)
            print("____________________________________")
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
            },
            status=status.HTTP_200_OK
        )
    
