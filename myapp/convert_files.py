# from django.shortcuts import render
# from django.http import HttpResponse
# from . import convert_files  # Import the convert_files module

# def home(request):
#     return render(request, 'myapp/home.html')

# def convert_files_view(request):
#     if request.method == 'POST':
#         input_folder = 'C:/Users/user/Downloads/Poulina PRJ/Invoices/Untreated'
#         output_folder = 'C:/Users/user/Downloads/Poulina PRJ/Invoices/Inputs'
#         convert_files.convert_and_resize_images(input_folder, output_folder)
#         return HttpResponse("Files converted and resized successfully!")
#     return HttpResponse("Conversion failed!")
