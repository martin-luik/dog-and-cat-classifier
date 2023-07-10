import os
import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model('./models/dog_and_cat_classifier_model_v17.h5')

input_data_dir = './input-images'
sorted_data_dir = './sorted-images'


for filename in os.listdir(input_data_dir):
    image_path = os.path.join(input_data_dir, filename)

    image = Image.open(image_path)
    image = image.resize((512, 512))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values to [0,1]

    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])
    class_labels = ['cat', 'dog']
    predicted_class_label = class_labels[predicted_class_index]

    if predicted_class_label == 'cat':
        print(f"{filename} is a cat.")
        os.rename(image_path, os.path.join(sorted_data_dir, 'cats', filename))
    elif predicted_class_label == 'dog':
        print(f"{filename} is a dog.")
        os.rename(image_path, os.path.join(sorted_data_dir, 'dogs', filename))
    else:
        print(f"{filename} is unknown.")
        os.rename(image_path, os.path.join(sorted_data_dir, 'unknown', filename))
