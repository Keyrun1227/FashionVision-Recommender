import streamlit as st
import pandas as pd
import os
from PIL import Image
import tensorflow
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors

# Set the page configuration to use the full screen width
st.set_page_config(layout="wide", page_title="🤖Kiran's FashionVision Recommender🔍️")


# Set the title and header
st.markdown("<h1 style='text-align: center; color: #ff5733; font-family: Times New Roman;'>FashionVision Recommender🛍️</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-family: Times New Roman;'>Find similar products to your uploaded or captured image 📸</h2>", unsafe_allow_html=True)

# Load the feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the CSV file
csv_data = pd.read_csv('styles.csv')

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    GlobalMaxPooling2D()
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

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def display_recommendations(indices, csv_data, filenames):
    st.subheader("Similar Products:")
    for row in range(2):
        cols_row = st.columns(3)
        for col in range(3):
            idx = row * 3 + col + 1  # Start from index 1
            if idx < len(indices[0]):
                product_info = csv_data.iloc[indices[0][idx]]
                with cols_row[col]:
                    st.write(f"<h4 style='color:red;'>{product_info['productDisplayName']}</h4>", unsafe_allow_html=True)
                    st.image(filenames[indices[0][idx]], width=200, caption=product_info['productDisplayName'])
                    st.markdown("  \n  \n")
                    st.write(f"**Gender:** {product_info['gender']}")
                    st.write(f"**Master Category:** {product_info['masterCategory']}")
                    st.write(f"**Sub Category:** {product_info['subCategory']}")
                    st.write(f"**Article Type:** {product_info['articleType']}")
                    st.write(f"**Base Colour:** {product_info['baseColour']}")
                    st.write(f"**Season:** {product_info['season']}")
                    st.write(f"**Year:** {product_info['year']}")
                    st.write(f"**Usage:** {product_info['usage']}")
                    st.write('---')


# Create the file uploader and camera input in separate columns
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an Image to get similar products", type=['jpg', 'png', 'jpeg'])

with col2:
    st.subheader("Capture an Image to get similar products")
    st.markdown('<style>div.Widget.row-widget.stRadio>div{flex-direction:row;}</style>',unsafe_allow_html=True)
    st.markdown('<style>#vgip_container {height:300px !important;}</style>', unsafe_allow_html=True)
    captured_image = st.camera_input("Capture")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.subheader("Uploaded Product:")
        st.image(Image.open(uploaded_file), width=300)
        features = feature_extractor(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)
        display_recommendations(indices, csv_data, filenames)
    else:
        st.header("Some Error Occurred in file upload")

if captured_image is not None:
    st.subheader("Captured Product:")
    st.image(captured_image, width=300)
    features = feature_extractor(captured_image, model)
    indices = recommend(features, feature_list)
    display_recommendations(indices, csv_data, filenames)
