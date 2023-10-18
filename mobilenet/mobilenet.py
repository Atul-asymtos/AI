import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Define a route to handle image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if the POST request contains an image file
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']

    # Check if the file has an allowed extension (e.g., .jpg or .png)
    if not file or file.filename == '':
        return jsonify({'error': 'Invalid file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Read and preprocess the uploaded image
        img_bytesio = io.BytesIO(file.read())
        img = image.load_img(img_bytesio, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make predictions on the image
        predictions = model.predict(x)

        # Decode and return the top-5 predicted classes
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        results = [{'class': label, 'score': float(score)} for (_, label, score) in decoded_predictions]

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

def allowed_file(filename):
    # Check if the file extension is allowed (you can customize this)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

