from django.urls import path
from .views import PredictView, UserFeedbackView


urlpatterns = [
    path('predict/', PredictView.as_view(), name='prediction'),
    path('feedback/<int:id_predict>/', UserFeedbackView.as_view(), name='feedback'),
]