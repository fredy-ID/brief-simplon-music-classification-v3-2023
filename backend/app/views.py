from .serializers import PredictSerializer, UserFeedbackSerializer, RetrainingSerializer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
from rest_framework.permissions import AllowAny
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
try:
    scaler = joblib.load("./app/ai_models/test/stable_model/scaler.pkl")
except:
    scaler = StandardScaler()
scaler2 = joblib.load("./app/ai_models_old_stable_model/scaler.pkl") 


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

    return segments, sample_rate


# try:
#     model = load_model('app/ai_models/stable_model')
# except Exception as e:
#     print('_____________________________')
#     print('Erreur lors du chargement du modèle :', e)
#     print('_____________________________')
#     raise

def audio_pipeline(audio, sample_rate): 
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

    # tempo = librosa.beat.tempo(y=audio)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
    features.append(tempo)

    # Calcul des moyennes des MFCC
    mfcc = librosa.feature.mfcc(y=audio)

    for x in mfcc:
        features.append(np.mean(x))
        features.append(np.var(x))

    return features, mfcc

class PredictView(generics.CreateAPIView):
    serializer_class = PredictSerializer
    permission_classes = (AllowAny,)
    
    
    def post(self, request, *args, **kwargs):
        toUserModel = request.data['modele']
        music = request.FILES['music']  # Suppose que la musique est téléchargée en tant que fichier
        print(music)
        # return Response("music")
        # audio, sr = librosa.load(music, sr=None)
        
        
        audio, sample_rate = get_3sec_sample(music)  # Récupère un tableau de features pour chaque 3 seconde de la musique
        features, mfcc = audio_pipeline(audio[2], sample_rate)  # Extrait les caractéristiques audio


        print('_________________________')
        print('_________________________')
        print("Nombre de caractéristiques extraites :", len(features))
        print('_________________________')
        print('_________________________')
        


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
            tempo = features[14],
            mfcc1_mean=np.mean(mfcc[0]),
            mfcc1_var=np.var(mfcc[0]),
            mfcc2_mean=np.mean(mfcc[1]),
            mfcc2_var=np.var(mfcc[1]),
            mfcc3_mean=np.mean(mfcc[2]),
            mfcc3_var=np.var(mfcc[2]),
            mfcc4_mean=np.mean(mfcc[3]),
            mfcc4_var=np.var(mfcc[3]),
            mfcc5_mean=np.mean(mfcc[4]),
            mfcc5_var=np.var(mfcc[4]),
            mfcc6_mean=np.mean(mfcc[5]),
            mfcc6_var=np.var(mfcc[5]),
            mfcc7_mean=np.mean(mfcc[6]),
            mfcc7_var=np.var(mfcc[6]),
            mfcc8_mean=np.mean(mfcc[7]),
            mfcc8_var=np.var(mfcc[7]),
            mfcc9_mean=np.mean(mfcc[8]),
            mfcc9_var=np.var(mfcc[8]),
            mfcc10_mean=np.mean(mfcc[9]),
            mfcc10_var=np.var(mfcc[9]),
            mfcc11_mean=np.mean(mfcc[10]),
            mfcc11_var=np.var(mfcc[10]),
            mfcc12_mean=np.mean(mfcc[11]),
            mfcc12_var=np.var(mfcc[11]),
            mfcc13_mean=np.mean(mfcc[12]),
            mfcc13_var=np.var(mfcc[12]),
            mfcc14_mean=np.mean(mfcc[13]),
            mfcc14_var=np.var(mfcc[13]),
            mfcc15_mean=np.mean(mfcc[14]),
            mfcc15_var=np.var(mfcc[14]),
            mfcc16_mean=np.mean(mfcc[15]),
            mfcc16_var=np.var(mfcc[15]),
            mfcc17_mean=np.mean(mfcc[16]),
            mfcc17_var=np.var(mfcc[16]),
            mfcc18_mean=np.mean(mfcc[17]),
            mfcc18_var=np.var(mfcc[17]),
            mfcc19_mean=np.mean(mfcc[18]),
            mfcc19_var=np.var(mfcc[18]),
            mfcc20_mean=np.mean(mfcc[19]),
            mfcc20_var=np.var(mfcc[19]),
        )

        x_t = np.array(features, dtype=float).reshape(1, -1)  # Transforme les caractéristiques en tableau 2D
        # return Response(x_t)
        # Échelonne les données
        
        
        print('_____________________________')
        print('toUserModel', int(toUserModel))
        print('_____________________________')
        try:
            if(int(toUserModel) == 2):
                x_t = scaler.transform(x_t)
                model = load_model('app/ai_models/test/stable_model')
            elif(int(toUserModel) == 1):
                x_t = scaler2.transform(x_t)
                model = load_model('app/ai_models_old_stable_model')
        except Exception as e:
            print('_____________________________')
            print('Erreur lors du chargement du modèle :', e)
            print('_____________________________')
            raise


        # Prédiction
        prediction = model.predict(x_t)
        # accuracy = accuracy_score(y_train, np.argmax(prediction, axis=1))

        probs = np.exp(prediction) / np.sum(np.exp(prediction), axis=1, keepdims=True)
        predicted_classes = np.argmax(probs, axis=1)
        predicted_class_names = [class_names[class_index] for class_index in predicted_classes]
        prediction_scores = probs.tolist()[0]
        print(f"Prédiction pour la musique : {predicted_class_names}")

        # instance = self.create(request, *args, **kwargs)

        predicted = Predict.objects.create(feature = savedFeature, prediction = predicted_class_names[0])

        predictedSerialized = PredictSerializer(predicted)
        return Response(            {
                'msg': "Prédiction faite",
                'predicted_classes': predicted_class_names,
                'prediction_scores': prediction_scores,
                'id': predictedSerialized.data.get("id"),
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
                "tempo": prediction.feature.tempo,
                "mfcc1_mean": prediction.feature.mfcc1_mean,
                "mfcc1_var": prediction.feature.mfcc1_var,
                "mfcc2_mean": prediction.feature.mfcc2_mean,
                "mfcc2_var": prediction.feature.mfcc2_var,
                "mfcc3_mean": prediction.feature.mfcc3_mean,
                "mfcc3_var": prediction.feature.mfcc3_var,
                "mfcc4_mean": prediction.feature.mfcc4_mean,
                "mfcc4_var": prediction.feature.mfcc4_var,
                "mfcc5_mean": prediction.feature.mfcc5_mean,
                "mfcc5_var": prediction.feature.mfcc5_var,
                "mfcc6_mean": prediction.feature.mfcc6_mean,
                "mfcc6_var": prediction.feature.mfcc6_var,
                "mfcc7_mean": prediction.feature.mfcc7_mean,
                "mfcc7_var": prediction.feature.mfcc7_var,
                "mfcc8_mean": prediction.feature.mfcc8_mean,
                "mfcc8_var": prediction.feature.mfcc8_var,
                "mfcc9_mean": prediction.feature.mfcc9_mean,
                "mfcc9_var": prediction.feature.mfcc9_var,
                "mfcc10_mean": prediction.feature.mfcc10_mean,
                "mfcc10_var": prediction.feature.mfcc10_var,
                "mfcc11_mean": prediction.feature.mfcc11_mean,
                "mfcc11_var": prediction.feature.mfcc11_var,
                "mfcc12_mean": prediction.feature.mfcc12_mean,
                "mfcc12_var": prediction.feature.mfcc12_var,
                "mfcc13_mean": prediction.feature.mfcc13_mean,
                "mfcc13_var": prediction.feature.mfcc13_var,
                "mfcc14_mean": prediction.feature.mfcc14_mean,
                "mfcc14_var": prediction.feature.mfcc14_var,
                "mfcc15_mean": prediction.feature.mfcc15_mean,
                "mfcc15_var": prediction.feature.mfcc15_var,
                "mfcc16_mean": prediction.feature.mfcc16_mean,
                "mfcc16_var": prediction.feature.mfcc16_var,
                "mfcc17_mean": prediction.feature.mfcc17_mean,
                "mfcc17_var": prediction.feature.mfcc17_var,
                "mfcc18_mean": prediction.feature.mfcc18_mean,
                "mfcc18_var": prediction.feature.mfcc18_var,
                "mfcc19_mean": prediction.feature.mfcc19_mean,
                "mfcc19_var": prediction.feature.mfcc19_var,
                "mfcc20_mean": prediction.feature.mfcc20_mean,
                "mfcc20_var": prediction.feature.mfcc20_var
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
        "tempo": [],
        "mfcc1_mean": [],
        "mfcc1_var": [],
        "mfcc2_mean": [],
        "mfcc2_var": [],
        "mfcc3_mean": [],
        "mfcc3_var": [],
        "mfcc4_mean": [],
        "mfcc4_var": [],
        "mfcc5_mean": [],
        "mfcc5_var": [],
        "mfcc6_mean": [],
        "mfcc6_var": [],
        "mfcc7_mean": [],
        "mfcc7_var": [],
        "mfcc8_mean": [],
        "mfcc8_var": [],
        "mfcc9_mean": [],
        "mfcc9_var": [],
        "mfcc10_mean": [],
        "mfcc10_var": [],
        "mfcc11_mean": [],
        "mfcc11_var": [],
        "mfcc12_mean": [],
        "mfcc12_var": [],
        "mfcc13_mean": [],
        "mfcc13_var": [],
        "mfcc14_mean": [],
        "mfcc14_var": [],
        "mfcc15_mean": [],
        "mfcc15_var": [],
        "mfcc16_mean": [],
        "mfcc16_var": [],
        "mfcc17_mean": [],
        "mfcc17_var": [],
        "mfcc18_mean": [],
        "mfcc18_var": [],
        "mfcc19_mean": [],
        "mfcc19_var": [],
        "mfcc20_mean": [],
        "mfcc20_var": [],
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
        limit = request.data['limit']
        
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
                    "tempo",
                    "mfcc1_mean",
                    "mfcc1_var",
                    "mfcc2_mean",
                    "mfcc2_var",
                    "mfcc3_mean",
                    "mfcc3_var",
                    "mfcc4_mean",
                    "mfcc4_var",
                    "mfcc5_mean",
                    "mfcc5_var",
                    "mfcc6_mean",
                    "mfcc6_var",
                    "mfcc7_mean",
                    "mfcc7_var",
                    "mfcc8_mean",
                    "mfcc8_var",
                    "mfcc9_mean",
                    "mfcc9_var",
                    "mfcc10_mean",
                    "mfcc10_var",
                    "mfcc11_mean",
                    "mfcc11_var",
                    "mfcc12_mean",
                    "mfcc12_var",
                    "mfcc13_mean",
                    "mfcc13_var",
                    "mfcc14_mean",
                    "mfcc14_var",
                    "mfcc15_mean",
                    "mfcc15_var",
                    "mfcc16_mean",
                    "mfcc16_var",
                    "mfcc17_mean",
                    "mfcc17_var",
                    "mfcc18_mean",
                    "mfcc18_var",
                    "mfcc19_mean",
                    "mfcc19_var",
                    "mfcc20_mean",
                    "mfcc20_var",
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
            # limit = 1
            # y = final_df['genre']
            # x = final_df.drop(columns=['genre'])
            genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            genre_counts = final_df['genre'].value_counts().to_dict()
            genres_used_for_training = []
            limited_data = []
            
            for genre, count in genre_counts.items():
                limited_genre_data = final_df[final_df['genre'] == genre].head(limit)
                limited_data.append(limited_genre_data)
                genres_used_for_training.extend([genre] * len(limited_genre_data))
                genre_counts[genre] = min(limit, len(limited_genre_data))

            limited_final_df = pd.concat(limited_data, axis=0)
            y = limited_final_df['genre']
            x = limited_final_df.drop(columns=['genre'])
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

            
            try:
                model.save("app/ai_models/test/stable_model/")
                model.save_weights('app/ai_models/test/stable_model/')
                joblib.dump(scaler, "app/ai_models/test/stable_model/scaler.pkl") 
            except Exception as e:
                print("____________________________________")
                print("le modèle n'a pas pu être sauvegardé")
                print(e)
                print("____________________________________")
                
            
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
        
        print("____________________________________")
        print("Enregistrement")
        print("____________________________________")
        
        self.create(request, *args, **kwargs)
        
        print("____________________________________")
        print("Enregistré")
        print("____________________________________")
        
        num_train_samples = len(x_train)
        num_test_samples = len(x_test)
        genres_used_for_training = genres[:num_train_samples]
        # genre_counts = final_df['genre'].value_counts().to_dict()
        # num_train_samples = limit * len(genres)
        # genres_used_for_training = genres[:limit] * len(genres)
    
        return Response(
            {
                'msg': "L'entraînement a été effectué",
                'epochs': epochs,
                'accuracy': accuracy,
                'num_train_samples': num_train_samples,
                'num_test_samples': num_test_samples,
                'genres_used_for_training': genres_used_for_training,
                'genre_counts': genre_counts
            },
            status=status.HTTP_200_OK
        )
        
        num_train_samples