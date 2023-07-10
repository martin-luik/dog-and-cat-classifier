import tensorflow as tf

train_dir = './dataset/training'
validation_dir = './dataset/validation'
test_dir = './dataset/test'

TARGET_IMG_SIZE = (512, 512)
BATCH_SIZE = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=TARGET_IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_data = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=TARGET_IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=TARGET_IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_data,
    epochs=20,
    validation_data=validation_data
)

loss, accuracy = model.evaluate(test_data)
print('Test accuracy:', accuracy)
print('Test loss:', loss)

model.save('./models/dog_and_cat_classifier_model_v1.h5')
