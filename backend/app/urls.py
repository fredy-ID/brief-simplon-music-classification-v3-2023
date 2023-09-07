from django.urls import path
from .views import PredictView, UserFeedbackView, RetrainingView


urlpatterns = [
    path('predict/', PredictView.as_view(), name='prediction'),
    path('feedback/<int:id_predict>/', UserFeedbackView.as_view(), name='feedback'),
    path('train-model/', RetrainingView.as_view(), name='train-model'),
]