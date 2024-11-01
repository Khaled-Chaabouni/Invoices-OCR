from django import forms

class UploadFileForm(forms.Form):
    invoiceFiles = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
