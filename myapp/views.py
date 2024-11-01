from pathlib import Path
from django.http import HttpResponse
import pandas as pd
from datetime import datetime
import os
from django.shortcuts import render
from django.conf import settings
from .utils import run_utils, predict_from_date


# Home page view
def home(request):
    return render(request, 'myapp/home.html')

# Features page view
def features(request):
    # Directory where detected images are stored
    detected_images_dir = settings.BASE_DIR / 'media' / 'Invoices' / 'Detected'
    
    # List of image URLs
    detected_images = []
    if detected_images_dir.exists():
        for filename in detected_images_dir.iterdir():
            if filename.suffix.lower() in ['.jpg', '.png']:  # Adjust for your image types
                image_url = settings.MEDIA_URL + f'Invoices/Detected/{filename.name}'
                detected_images.append({
                    'url': image_url,
                    'name': filename.name
                })
    
    return render(request, 'myapp/features.html', {
        'detected_images': detected_images
    })

# Run utils view
def run_utils_view(request):
    try:
        # Run all utility functions
        run_utils()

        # Load the generated images from the Detected folder
        detected_images = []
        detected_folder = settings.BASE_DIR / 'media' / 'Detected'
        
        if detected_folder.exists():
            for filename in detected_folder.iterdir():
                if filename.suffix.lower() in ['.jpg', '.png']:
                    detected_images.append({
                        'url': settings.MEDIA_URL + f'Detected/{filename.name}'
                    })

        # Read the dataset.csv file and convert it to HTML
        dataset_path = settings.BASE_DIR / 'Extracted Data' / 'dataset.csv'  # Adjust path as needed
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            dataset_html = df.to_html(classes="table table-striped table-bordered", index=False)
        else:
            dataset_html = "Dataset not found."

        # Render the features template with the dataset and detected images
        return render(request, 'myapp/features.html', {
            'dataset_html': dataset_html,
            'detected_images': detected_images
        })

    except Exception as e:
        return HttpResponse(f"An error occurred: {str(e)}")
    
def predict_invoice(request):
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            date_str = request.POST.get('date')
            # Validate and parse the date
            date = datetime.strptime(date_str, "%d/%m/%Y")

            # Call the prediction function with the date formatted as a string
            prediction = predict_from_date(date_str)  # Pass date_str directly

        except ValueError:
            error_message = "Invalid date format. Please use dd/mm/yyyy."
        except Exception as e:
            error_message = str(e)

    return render(request, 'myapp/prediction.html', {'prediction': prediction, 'error_message': error_message})
