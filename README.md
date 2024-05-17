# 🛍️ FashionVision Recommender 📸
![Image Alt Text](https://github.com/Keyrun1227/FashionVision-Recommender/blob/main/cam1.png)

## Project Overview
The **FashionVision Recommender** is a powerful image-based recommendation system that helps users find similar fashion products based on an uploaded or captured image. This project utilizes the Fashion Product Images (Small) dataset from Kaggle to train a deep learning model and provide personalized recommendations to users.

## 📚 Dataset Overview
The [Fashion Product Images (Small) dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) consists of 44,000 fashion products with category labels and images. Each product is identified by a unique ID, and the dataset includes the following information:

- **Product Images**: The images for each product are stored in the `images/` directory, with the file name following the pattern `{product_id}.jpg`.
- **Product Metadata**: The metadata for each product is stored in the `styles.csv` file, which includes information such as the product's category, gender, color, season, and more.

## 🛠️ Project Structure
The project consists of the following main components:

1. **Fashion_recommendation.py**: This file is responsible for extracting features from the product images using a pre-trained ResNet50 model. The extracted features are then saved to a file for later use.
2. **testing.py**: This file demonstrates how to use the extracted features to find similar products for a given image. It loads the saved features and filenames, and then uses a nearest-neighbor algorithm to recommend the top similar products.
3. **app.py**: This is the main Streamlit application that provides the user interface for the Fashion Vision Recommender. It allows users to upload or capture an image, and then displays the similar products based on the extracted features.

## 🚀 How It Works
1. **Feature Extraction**: The `Fashion_recommendation.py` script extracts features from the 44,441 product images using a pre-trained ResNet50 model. This process takes approximately 3 hours and 55 minutes to complete, as the model needs to process each image. The extracted features are then saved to a file for later use.
2. **Recommendation Engine**: The `testing.py` script loads the saved features and filenames, and then uses a nearest-neighbor algorithm to find the top 5 similar products for a given input image. This process is relatively fast, as it only needs to compare the input image's features with the pre-computed features.
3. **Streamlit Application**: The `app.py` script brings everything together in a user-friendly Streamlit application. Users can either upload an image or capture one using their device's camera. The application then displays the similar products, including their product details and images.

## 💡 Key Features
- **Image-based Recommendations**: The system provides personalized recommendations based on the visual similarity of the input image to the products in the dataset.
- **Efficient Feature Extraction**: The use of a pre-trained ResNet50 model for feature extraction ensures that the recommendations are accurate and fast.
- **User-friendly Interface**: The Streamlit application provides a seamless and intuitive experience for users to find similar fashion products.
- **Comprehensive Product Information**: The recommended products are displayed with detailed metadata, including the product name, gender, category, color, and more.

## 🌟 Real-life Applications
The FashionVision Recommender can be highly useful in various real-life scenarios, such as:

- **E-commerce Personalization**: Online fashion retailers can integrate this system into their platforms, allowing customers to easily find similar products based on their interests and preferences.
- **Fashion Inspiration and Discovery**: Individuals can use the system to discover new fashion trends and styles, helping them expand their wardrobe and explore different fashion options.
- **Retail Store Assistance**: Brick-and-mortar fashion stores can leverage this system to provide personalized recommendations to customers, enhancing their shopping experience and increasing sales.
- **Fashion Design and Trend Analysis**: Fashion designers and industry professionals can use the system to analyze product trends, identify popular styles, and gain insights for their future collections.

## 🤖 Future Enhancements
To further improve the FashionVision Recommender, some potential future enhancements include:

- **Integration with User Preferences**: Incorporating user feedback and past purchase history to personalize the recommendations and improve the system's accuracy.
- **Multi-modal Recommendations**: Combining image-based and text-based (e.g., product descriptions) features to provide more comprehensive recommendations.
- **Scalable Infrastructure**: Optimizing the feature extraction process and deploying the system on a scalable infrastructure to handle large-scale real-time requests.
- **Deployment as a Web Service**: Developing the system as a standalone web service that can be easily integrated into various e-commerce platforms and applications.

## 📚 Resources
- [Fashion Product Images (Small) dataset on Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- [ResNet50 model documentation](https://keras.io/api/applications/resnet/#resnet50-function)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Nearest Neighbors algorithm documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)

## FashionVision Recommender - User Interaction Screenshots
![Image Alt Text](https://github.com/Keyrun1227/FashionVision-Recommender/blob/main/cam2.png)

## FashionVision - Image Upload Interface
![Image Alt Text](https://github.com/Keyrun1227/FashionVision-Recommender/blob/main/up1.png)
![Image Alt Text](https://github.com/Keyrun1227/FashionVision-Recommender/blob/main/up2.png)

## Flask Implementation
In addition to the Streamlit-based application, the FashionVision Recommender project has also been implemented using the Flask web framework. The Flask-based application provides a similar user interface and functionality as the Streamlit version, allowing users to upload or capture images and receive personalized fashion recommendations.

## FashionVision - Flask
![Image Alt Text](https://github.com/Keyrun1227/FashionVision-Recommender/blob/main/flask.png)

The Flask application is contained in the `flask/` folder of the project repository. The main components of the Flask implementation are:

1. **app.py**: This is the main Flask application file that handles the routing and rendering of the user interface.
2. **recommendation.py**: This file contains the logic for extracting features from the product images and finding the most similar products.
3. **templates/**: This directory contains the HTML templates used to render the user interface.
4. **static/**: This directory holds the CSS files and other static assets used by the Flask application.

To run the Flask-based FashionVision Recommender, you will need to have Flask installed. You can install it using pip:

```bash
pip install flask
```
Once Flask is installed, you can run the Flask-based FashionVision Recommender by navigating to the flask/ directory in your terminal and executing the following command:
```bash
python app.py
```
This will start the Flask server, and you can then access the FashionVision Recommender application by opening a web browser and navigating to http://localhost:5000.

## 🤝 Contribution
If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request on the project's [FashionVision-Recommender](https://github.com/Keyrun1227/FashionVision-Recommender).

## 🎉 Conclusion
The FashionVision Recommender is a powerful tool that can revolutionize the way people discover and shop for fashion products. By leveraging the latest advancements in computer vision and deep learning, this project provides a seamless and personalized recommendation experience, empowering users to find their perfect fashion matches. 🛍️👗👔
