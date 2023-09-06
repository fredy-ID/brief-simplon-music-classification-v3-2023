from django import forms


class UploadFileForm(forms.Form):
    music = forms.FileField()