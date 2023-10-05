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
    
    mfcc1_mean = models.FloatField()
    mfcc1_var = models.FloatField()
    mfcc2_mean = models.FloatField()
    mfcc2_var = models.FloatField()
    mfcc3_mean = models.FloatField()
    mfcc3_var = models.FloatField()
    mfcc4_mean = models.FloatField()
    mfcc4_var = models.FloatField()
    mfcc5_mean = models.FloatField()
    mfcc5_var = models.FloatField()
    mfcc6_mean = models.FloatField()
    mfcc6_var = models.FloatField()
    mfcc7_mean = models.FloatField()
    mfcc7_var = models.FloatField()
    mfcc8_mean = models.FloatField()
    mfcc8_var = models.FloatField()
    mfcc9_mean = models.FloatField()
    mfcc9_var = models.FloatField()
    mfcc10_mean = models.FloatField()
    mfcc10_var = models.FloatField()
    mfcc11_mean = models.FloatField()
    mfcc11_var = models.FloatField()
    mfcc12_mean = models.FloatField()
    mfcc12_var = models.FloatField()
    mfcc13_mean = models.FloatField()
    mfcc13_var = models.FloatField()
    mfcc14_mean = models.FloatField()
    mfcc14_var = models.FloatField()
    mfcc15_mean = models.FloatField()
    mfcc15_var = models.FloatField()
    mfcc16_mean = models.FloatField()
    mfcc16_var = models.FloatField()
    mfcc17_mean = models.FloatField()
    mfcc17_var = models.FloatField()
    mfcc18_mean = models.FloatField()
    mfcc18_var = models.FloatField()
    mfcc19_mean = models.FloatField()
    mfcc19_var = models.FloatField()
    mfcc20_mean = models.FloatField()
    mfcc20_var = models.FloatField()

class Retraining(models.Model):
    id = models.BigAutoField(primary_key=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    
class CSVDataset(models.Model):
    id    = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=50)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    
class Predict(models.Model):
    id    = models.BigAutoField(primary_key=True)
    music = models.FileField(null=True)
    prediction = models.TextField(null=True)
    
    feature = models.ForeignKey(Features, on_delete=models.CASCADE)
    exists_in_csv = models.BooleanField(default=False)
    
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
    
