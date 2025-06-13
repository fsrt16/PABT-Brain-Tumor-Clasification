import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMAGE_SIZE = (224, 224)  # Set to match your model input size
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def load_images_from_directory(base_path):
    images = []
    labels = []
    for label in CLASS_LABELS:
        folder = os.path.join(base_path, label)
        if not os.path.exists(folder):
            print(f"Warning: Directory {folder} does not exist.")
            continue

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
    return np.array(images), np.array(labels)


def normalize_images(images):
    return images.astype('float32') / 255.0


def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels)
    return one_hot_labels, encoder


def split_dataset(images, labels, test_size=0.15, val_size=0.15):
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, stratify=labels, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, stratify=y_train, random_state=42)
    return x_train, x_val, x_test, y_train, y_val, y_test


def get_image_generators(x_train, y_train, x_val, y_val):
    train_aug = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_aug = ImageDataGenerator()  # Only rescaling for validation

    train_generator = train_aug.flow(x_train, y_train, batch_size=32)
    val_generator = val_aug.flow(x_val, y_val, batch_size=32)

    return train_generator, val_generator


def preprocess_pipeline(base_path):
    print("Loading images...")
    images, labels = load_images_from_directory(base_path)

    print("Normalizing images...")
    images = normalize_images(images)

    print("Encoding labels...")
    one_hot_labels, encoder = encode_labels(labels)

    print("Splitting dataset...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(images, one_hot_labels)

    print("Preparing data generators with augmentation...")
    train_generator, val_generator = get_image_generators(x_train, y_train, x_val, y_val)

    return train_generator, val_generator, x_test, y_test, encoder


if __name__ == "__main__":
    base_path = "dataset/brain_mri"  # Change to your dataset path
    train_gen, val_gen, x_test, y_test, label_encoder = preprocess_pipeline(base_path)
    print("Preprocessing complete. Ready for training.")
