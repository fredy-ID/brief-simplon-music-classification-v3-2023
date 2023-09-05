import os
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response


from .serializers import ScannerSerializer

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop','jazz', 'metal', 'pop', 'reggae', 'rock']
num_classes = len(class_names)

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

try:
    model = load_model('app/ai_models/modelF.hdf5')
except Exception as e:
    print('_____________________________')
    print('Erreur lors du chargement du modèle :', e)
    print('_____________________________')
    raise

class PredictView(generics.CreateAPIView):
    serializer_class = ScannerSerializer
    permission_classes = (AllowAny,)

    def create(self, request, *args, **kwargs):
        return Response(
            {
                'msg': "prédiction faite"
            }, 
            status=status.HTTP_200_OK
        )