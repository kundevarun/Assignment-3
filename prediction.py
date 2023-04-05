# Import necessary modules
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load and preprocess the image
img_path = 'V:\\Data Visualization\\Project\\Data\\test1.png'
img = load_img(img_path, target_size=(150, 150))
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
prediction = model.predict(img_array)

print(prediction)

# Print the prediction
if prediction[0][0] > 0.5:
    print('The image is classified as a human.')
else:
    print('The image is classified as a car.')
