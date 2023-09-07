from django.db import models


class Predict(models.Model):
    id = models.BigAutoField(primary_key=True)
    music = models.FileField(null=True)

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