from django.db import models

class Features(models.Model):
    id = models.BigAutoField(primary_key=True)
    
    chroma_stft_mean = models.FloatField()
    chroma_stft_var = models.FloatField()

    rms_mean = models.FloatField()
    rms_var = models.FloatField()

    spectral_centroids_mean = models.FloatField()
    spectral_centroids_var = models.FloatField()

    spectral_bandwidth_mean = models.FloatField()
    spectral_bandwidth_var = models.FloatField()

    rolloff_mean = models.FloatField()
    rolloff_var = models.FloatField()

    zcr_mean = models.FloatField()
    zcr_var = models.FloatField()

    harmony_mean = models.FloatField()
    harmony_var = models.FloatField()

    tempo_mean = models.FloatField()
    tempo_var = models.FloatField()

    
class Predict(models.Model):
    id    = models.BigAutoField(primary_key=True)
    music = models.FileField(null=True)
    feature = models.OneToOneField(Features, on_delete=models.CASCADE)
    prediction = models.TextField(null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
class UserFeedback(models.Model):
    GENRE_CHOICES = [
        ('blues', 'Blues'),
        ('classical', 'Classical'),
        ('country', 'Country'),
        ('disco', 'Disco'),
        ('hiphop', 'Hip Hop'),
        ('jazz', 'Jazz'),
        ('metal', 'Metal'),
        ('pop', 'Pop'),
        ('reggae', 'Reggae'),
        ('rock', 'Rock'),
    ]
    
    id = models.BigAutoField(primary_key=True)
    genre_feedback = models.CharField(max_length=30, choices=GENRE_CHOICES)
    
    predict = models.ForeignKey(Predict, on_delete=models.DO_NOTHING)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)