import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from glob import glob
import random
from PIL import Image

def load_dataset(image_dir, image_exts=['jpg', 'jpeg', 'png']):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(image_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls in class_names:
        cls_folder = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        for ext in image_exts:
            for img_path in glob(os.path.join(cls_folder, f"*.{ext}")):
                image_paths.append(img_path)
                labels.append(class_to_idx[cls])
    
    return np.array(image_paths), np.array(labels), class_to_idx


def plot_class_distribution(y, title='Class Distribution'):
    counter = Counter(y)
    labels, values = zip(*sorted(counter.items()))
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(labels), y=list(values))
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Samples')
    plt.tight_layout()
    plt.show()


def display_sample_images(X, y, class_to_idx, num_samples=3):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    unique_classes = np.unique(y)
    plt.figure(figsize=(15, 5))
    
    for class_id in unique_classes:
        indices = np.where(y == class_id)[0]
        selected_indices = np.random.choice(indices, size=min(num_samples, len(indices)), replace=False)
        for i, idx in enumerate(selected_indices):
            img = Image.open(X[idx])
            plt.subplot(len(unique_classes), num_samples, class_id * num_samples + i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{idx_to_class[class_id]}")
    plt.tight_layout()
    plt.show()


def split_dataset(X, y, test_size=0.2, val_size=None, stratify=True, random_state=42):
    stratify_option = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_option, random_state=random_state
    )

    if val_size:
        stratify_option = y_train if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=stratify_option, random_state=random_state
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_test, y_test


# ====================== Example Usage =======================

if __name__ == "__main__":
    image_directory = "dataset_path"  # Replace with your dataset path, e.g., "./data"

    # Load image paths and labels
    X, y, class_map = load_dataset(image_directory)

    print(f"Total samples: {len(X)}")
    print(f"Class mapping: {class_map}")

    # Perform train/test/val split
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y, test_size=0.15, val_size=0.15, stratify=True
    )

    # Print class distributions
    print("\nTrain Class Distribution:")
    plot_class_distribution(y_train, title='Train Set')
    print("Validation Class Distribution:")
    plot_class_distribution(y_val, title='Validation Set')
    print("Test Class Distribution:")
    plot_class_distribution(y_test, title='Test Set')

    # Display some sample images from the training set
    display_sample_images(X_train, y_train, class_map, num_samples=3)
