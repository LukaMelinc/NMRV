import os
import shutil
import random
from sklearn.model_selection import train_test_split

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_dataset(image_dir, annotation_dir, train_image_dir, val_image_dir, train_annotation_dir, val_annotation_dir, val_split=0.2):
    # Get the list of all images
    images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # Split the dataset into training and validation sets
    train_images, val_images = train_test_split(images, test_size=val_split, random_state=42)

    # Create directories for training and validation sets
    create_directory(train_image_dir)
    create_directory(val_image_dir)
    create_directory(train_annotation_dir)
    create_directory(val_annotation_dir)

    # Move images and annotations to the appropriate directories
    for image in train_images:
        src_image_path = os.path.join(image_dir, image)
        dest_image_path = os.path.join(train_image_dir, image)
        shutil.move(src_image_path, dest_image_path)

        src_annotation_path = os.path.join(annotation_dir, image.replace('.jpg', '.txt'))
        dest_annotation_path = os.path.join(train_annotation_dir, image.replace('.jpg', '.txt'))
        if os.path.exists(src_annotation_path):
            shutil.move(src_annotation_path, dest_annotation_path)

    for image in val_images:
        src_image_path = os.path.join(image_dir, image)
        dest_image_path = os.path.join(val_image_dir, image)
        shutil.move(src_image_path, dest_image_path)

        src_annotation_path = os.path.join(annotation_dir, image.replace('.jpg', '.txt'))
        dest_annotation_path = os.path.join(val_annotation_dir, image.replace('.jpg', '.txt'))
        if os.path.exists(src_annotation_path):
            shutil.move(src_annotation_path, dest_annotation_path)

    print(f"Dataset split complete. {len(train_images)} training images and {len(val_images)} validation images.")

# Example usage
image_dir = '/content/NMRV/Seminar/RGB_images/train'
annotation_dir = '/content/NMRV/Seminar/RGB_annotations/train'
train_image_dir = '/content/NMRV/Seminar/RGB_images/train'
val_image_dir = '/content/NMRV/Seminar/RGB_images/val'
train_annotation_dir = '/content/NMRV/Seminar/RGB_annotations/train'
val_annotation_dir = '/content/NMRV/Seminar/RGB_annotations/val'

split_dataset(image_dir, annotation_dir, train_image_dir, val_image_dir, train_annotation_dir, val_annotation_dir, val_split=0.2)
