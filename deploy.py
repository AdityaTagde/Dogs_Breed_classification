import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model (ensure the model is in the correct directory)
model = load_model('dog_breed_model.h5')

# Define directories and allowed file extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route: Displays the HTML form to upload an image
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route: Handles image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part'
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the uploaded image
        img = image.load_img(filepath, target_size=(140, 140))  # Resize as per model input
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        # Get class names (the names of the dog breeds from your dataset)
        class_names = ['Boxer', 'Bulldog', 'Chihuahua', 'Dachshund', 'German Shepherd', 'Golden Retriever', 'Labrador', 'Poodle', 'Rottweiler', 'Shih Tzu']  # Example list, replace with your actual list
        breed_name = class_names[predicted_class[0]]

        # Return the result as HTML
        return f'''
            <html>
            <body>
                <h1>Prediction Result</h1>
                <p>Predicted breed: {breed_name}</p>
                <p><img src="/static/uploads/{filename}" width="300" alt="Uploaded Image"></p>
                <a href="/">Go back to home</a>
            </body>
            </html>
        '''

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
