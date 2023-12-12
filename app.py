from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)

# Load the pre-trained skin cancer classification model
skin_model = load_model(
    '/Users/coyscoyscoys/Downloads/CM Project/full_skin_cancer_model.h5')

# Class names
class_names = [
    'melanocytic nevi', 'melanoma', 'basal cell carcinoma',
    'Actinic keratoses and intraepithelial carcinoma',
    'vascular lesions', 'benign keratosis-like', 'dermatofibroma'
]

# Define function to preprocess image


def preprocess_image(img):
    processed_img = image.load_img(img, target_size=(224, 224))
    processed_img = image.img_to_array(processed_img)
    processed_img = np.expand_dims(processed_img, axis=0)
    processed_img = processed_img / 255.0  # Normalize pixel values
    return processed_img

# Define route for model inference


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = preprocess_image(file)
        prediction = skin_model.predict(img)
        # Convert prediction to human-readable format using class names
        result = {class_names[i]: float(prediction[0][i])
                  for i in range(len(class_names))}
        return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
