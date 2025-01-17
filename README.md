
# 🐶 Dog Breed Classifier 🐕

This project is a Flask web application that allows users to upload an image of a dog and predict its breed using a pre-trained machine learning model. The model is built using TensorFlow and Keras, and it classifies dog breeds based on the uploaded image.


## 📑 Table of contents
- Project Overview
- Installation
- Usage
- File Structure
- Dependencies
- Contributing


## Project overview
This project uses a pre-trained deep learning model to classify dog breeds. When a user uploads an image of a dog, the application processes the image and returns a prediction of the dog’s breed. The app is built using Flask for the backend and TensorFlow for the machine learning model, with HTML.
## ⚡ Usage
1. Start the Flask server:
bash
Copy
python app.py
By default, the app will run locally on http://127.0.0.1:5000/.

2. Open the app in your browser:
Navigate to http://127.0.0.1:5000/ in your browser.

3. Upload an image:
Click the "Choose File" button to upload an image of a dog (JPEG, PNG, or JPG). The app will process the image and display the predicted breed.

4. View the result:
Once the image is processed, the app will display the predicted breed and show the image you uploaded.


## 🗂️ File Structure

Here’s an overview of the project’s directory structure:
dog-breed-classifier/
![Alter](https://github.com/AdityaTagde/Dogs_Breed_classification/blob/main/Screenshot%202025-01-17%20234833.png)

## 🧩 Dependencies
The following Python libraries are required to run the app:

- Flask: Web framework for Python.
- tensorflow: Deep learning framework for building and using the model.
- werkzeug: Provides utilities for handling file uploads.
- numpy: For handling arrays and image preprocessing.
- Pillow: For image processing and loading.
## 💡 Notes
- The app uses a simple image classification model that accepts images of size 140x140 pixels. If your model requires a different input size, make sure to update the target_size in the app.py file accordingly.
- The app uses static/uploads to save the uploaded images. Make sure the folder exists or is created when the app starts.

  
##🖼️ Screenshot

![App Screenshot](https://github.com/AdityaTagde/Dogs_Breed_classification/blob/main/Screenshot%202025-01-17%20234833.png)

![App Screenshot](https://github.com/AdityaTagde/Dogs_Breed_classification/blob/main/Screenshot%202025-01-17%20235417.png)

