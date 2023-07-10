import os
import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model('./models/dog_and_cat_classifier_model_v1.h5')

input_data_dir = './input-images'
sorted_data_dir = './sorted-images'
sorted_data_cats_dir = os.path.join(sorted_data_dir, 'cats')
sorted_data_dogs_dir = os.path.join(sorted_data_dir, 'dogs')

if not os.path.exists(sorted_data_cats_dir):
    os.makedirs(sorted_data_cats_dir)
if not os.path.exists(sorted_data_dogs_dir):
    os.makedirs(sorted_data_dogs_dir)

for filename in os.listdir(input_data_dir):
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(input_data_dir, filename)

    image = Image.open(image_path)
    image = image.resize((512, 512))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

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
