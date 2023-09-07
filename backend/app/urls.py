from django.urls import path
from .views import PredictView, UserFeedbackView, Dataset


urlpatterns = [
    path('predict/', PredictView.as_view(), name='prediction'),
    path('feedback/<int:id_predict>/', UserFeedbackView.as_view(), name='feedback'),
    path('predictions/', Dataset.as_view(), name="predictions")
]