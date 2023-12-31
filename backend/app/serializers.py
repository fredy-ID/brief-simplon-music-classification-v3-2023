from rest_framework import serializers
from .models import Predict, UserFeedback, Features, CSVDataset, Retraining

class PredictSerializer(serializers.ModelSerializer):
    class Meta:
        model = Predict
        fields = '__all__'

    def create(self, validated_data):
        """
        Create and return a new `Snippet` instance, given the validated data.
        """
        return Predict.objects.create(**validated_data)


class FeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Features
        fields = '__all__'


    def create(self, validated_data):
        """
        Create and return a new `Snippet` instance, given the validated data.
        """
        return Features.objects.create(**validated_data)

class UserFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserFeedback
        fields = '__all__'

class CSVDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = CSVDataset
        fields = '__all__'
        
class RetrainingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Retraining
        fields = '__all__'