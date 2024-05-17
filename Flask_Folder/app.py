from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the CSV file
csv_data = pd.read_csv('styles.csv')

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalMaxPooling2D()
])

# Functions to extract features and recommend similar products
def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalised_result = result / norm(result)
    return normalised_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

@app.route('/')
def home():
    return render_template('index.html')

import json

@app.route('/recommend', methods=['POST'])
def recommend_products():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Create the 'static/uploads' directory if it doesn't exist
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
        
        # Save the uploaded file
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        
        # Extract features of the uploaded image
        features = feature_extractor(file_path, model)
        
        # Recommend similar products
        indices = recommend(features, feature_list)
        
        # Prepare product information
        products = []
        for idx in indices[0]:
            product_info = csv_data.iloc[idx].to_dict()
            # Handle NaN values
            for key, value in product_info.items():
                if pd.isna(value):
                    product_info[key] = None  # Convert NaN to None
            # Replace backslashes with forward slashes in the image path
            product_info['image'] = f"static/images/{os.path.basename(filenames[idx])}"
            products.append(product_info)
        
        # Convert to JSON
        response_data = {'uploaded_image': f"static/uploads/{file.filename}", 'products': products}
        json_response = json.dumps(response_data)
        
        return json_response, 200, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    app.run(debug=True)
