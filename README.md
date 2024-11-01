# **Invoice Processing Tool**
![image](https://github.com/user-attachments/assets/1019aec6-83df-4f0c-9383-2b895908897f)


Overview
The Invoice Processing Tool is designed to automate the extraction and prediction of future invoice values from electricity bills. The application utilizes Optical Character Recognition (OCR) to parse data from invoice images and provides a user-friendly interface to interact with the processed data.

Features
Image Upload: Users can upload images of electricity invoices.
Data Extraction: The tool extracts relevant data from invoices using OCR.
Future Prediction: Predicts future invoice values based on historical data.
Data Visualization: Displays processed data in a user-friendly format.
![image](https://github.com/user-attachments/assets/9d17d101-6486-41d0-b859-2f7f37f1af44)


Technologies Used
Django
Python
Bootstrap
OpenCV
Tesseract OCR
Machine Learning (for prediction)
Getting Started
Prerequisites
Python 3.x
Django
Necessary Python packages (as specified in requirements.txt)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/Khaled-Chaabouni/Invoices-OCR.git
cd Invoices-OCR
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set up the database:

bash
Copy code
python manage.py migrate
Run the development server:

bash
Copy code
python manage.py runserver
Access the application at http://127.0.0.1:8000/.

Usage
Upload your electricity invoice images through the provided interface.
Click on the "Run Processing" button to start the data extraction process.
View the extracted data and predicted values directly on the application interface.
![image](https://github.com/user-attachments/assets/53c798df-c953-47b3-9c36-fb48aa049a14)


Deployed Application
The Invoice Processing Tool is also deployed and can be accessed online. Here are some screenshots of the live application:

![image](https://github.com/user-attachments/assets/1630ec12-a3ed-4dfc-a892-0f444124a8bd)

![image](https://github.com/user-attachments/assets/4664a1ac-d7f5-458d-a066-eea5dc884431)

![image](https://github.com/user-attachments/assets/838251b5-a483-42f1-9a8e-83e97c3d3dcb)



Contributing
Contributions are welcome! If you have suggestions for improvements or want to report issues, please submit a pull request or open an issue in this repository.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Make sure to replace the placeholder paths with actual paths to your images. This structure provides a comprehensive overview of your project while showcasing its features visually!
