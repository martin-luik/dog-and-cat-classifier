import tensorflow as tf

# Step 1: Load and preprocess the training, validation, and test data
train_dir = './dataset/training'
validation_dir = './dataset/validation'
test_dir = './dataset/test'

TARGET_IMG_SIZE = (512, 512)

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
    batch_size=32,
    class_mode='binary'
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_data = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=TARGET_IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=TARGET_IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)

# Step 2: Create the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model with validation
model.fit(
    train_data,
    epochs=20,
    validation_data=validation_data
)

# Step 5: Evaluate the model on the test data
loss, accuracy = model.evaluate(test_data)
print('Test accuracy:', accuracy)

model.save('./models/dog_and_cat_classifier_model_v1.h5')