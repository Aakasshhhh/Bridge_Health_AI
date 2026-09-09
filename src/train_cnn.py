import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import sys

train_dir = 'data/images/train'
val_dir = 'data/images/val'
MODEL_FILE_NAME = "cnn_model.h5"
MODEL_PATH = os.path.join("models", MODEL_FILE_NAME)

datagen = ImageDataGenerator(rescale=1./255)

print("📥 Preparing Image Data...")
try:
    train_data = datagen.flow_from_directory(
        train_dir, 
        target_size=(150, 150), 
        batch_size=32, 
        class_mode='binary' 
    )
    val_data = datagen.flow_from_directory(
        val_dir, 
        target_size=(150, 150), 
        batch_size=32, 
        class_mode='binary'
    )
except Exception as e:
    print(f"🚨 ERROR: Could not load image data. Check if your image files are correctly organized under 'data/images/'.")
    print(f"Error details: {e}")
    sys.exit() 

print("⚙️ Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(), 
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Training the CNN model for 10 epochs...")
print("This may take a few minutes...")
history = model.fit(
    train_data, 
    validation_data=val_data, 
    epochs=10
)

os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

print(f"\n✅ CNN model saved successfully!")
print(f"File location confirmed: {MODEL_PATH}")