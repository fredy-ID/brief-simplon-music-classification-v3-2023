# Generated by Django 4.2.5 on 2023-09-07 06:15

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_predict_music'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserFeedback',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('genre_feedback', models.CharField(choices=[('blues', 'Blues'), ('classical', 'Classical'), ('country', 'Country'), ('disco', 'Disco'), ('hiphop', 'Hip Hop'), ('jazz', 'Jazz'), ('metal', 'Metal'), ('pop', 'Pop'), ('reggae', 'Reggae'), ('rock', 'Rock')], max_length=30)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('predict', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='app.predict')),
            ],
        ),
    ]
