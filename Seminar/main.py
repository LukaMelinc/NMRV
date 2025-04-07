import torch
import json
import cv2
import os
from ultralytics import YOLO
from network import UNet
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from dataloader import *


def count_img(dri_path):
    count = 0
    for root, dirs, files in os.walk(dri_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                count += 1

    print(count)
    return count

# Predprocesiranje termalnih slik
def preprocess_thermal(thermal):

    thermal = cv2.cvtColor(thermal, cv2.COLOR_GRAY2RGB)
    
    thermal = cv2.normalize(thermal, dst=None, alpha=0, beta=65535,
    
    norm_type=cv2.NORM_MINMAX)
    
    thermal = cv2.convertScaleAbs(thermal, alpha=255/(2**16))
    
    return thermal

def load_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = []
    annotations = []

    for item in data:
        image_path = item['image_path']
        image = cv2.imread(image_path)
        
        if image is not None:
            images.append(image)
            annotations.append(item['annotations'])

    return images, annotations


def semantic_segmentation(image, model):
    image = cv2.resize(image(256, 256))
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32)
    with torch.no_grad():
        output = model(image)

    return output

def display_image_and_annotation(image, detection_resutls, segmentation_mask):

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Detection result
    plt.subplot(1, 3, 2)
    plt.title('Detection Results')
    plt.imshow(detection_resutls.plot())

    # Segmentation mask
    plt.subplot(1, 3, 3)
    plt.title('Segmentation Mask')
    plt.imshow(segmentation_mask.squeeze().cpu().numpy(), cmap='gray')

    plt.show()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_objects(image, model):
    # Loading the YOLO model

    result = model(image)

    return result

def calculate_F1(precision, recall):
    if precision + recall == 0:
        return 0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_precision(detection_result, annotations, iou_hreshold=0.5)_
    true_positive=0
    for detection, annotation in zip(detection_result, annotations):
        for pred in detection:
            for gt in annotation:
                iou = calculate_iou(pred, gt)
                if iou >= iou_hreshold:
                    true_positive += 1
                    break

    return true_positive / len(detection_result) if detection_result else 0

# Calculate recall
def calculate_recall(detection_results, annotations, iou_threshold=0.5):
    true_positives = 0
    for detection, annotation in zip(detection_results, annotations):
        for gt in annotation:
            for pred in detection:
                iou = calculate_iou(pred, gt)
                if iou >= iou_threshold:
                    true_positives += 1
                    break
    return true_positives / len(annotations) if annotations else 0


def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection / (box1_area + box2_area - intersection)

    return iou


def display_image_and_annotations(image, detection_result, segmentation_mask):
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Detection result
    plt.subplot(1, 3, 2)
    plt.title('Detection Results')
    plt.imshow(detection_result.plot())

    # Segmentation mask
    plt.subplot(1, 3, 3)
    plt.title('Segmentation Mask')
    plt.imshow(segmentation_mask.squeeze().cpu().numpy(), cmap='gray')

    plt.show()

def main():
    json_file = 'TODO'
    images, annotations = load_dataset(json_file)
    train_file = "train.txt"
    test_file = "test.txt"
    train_data, test_data = load_split_files(train_file, test_file)


    train_dataset = CustomDataloader(json_file, transform=transform, images=train_data)
    test_dataset = CustomDataloader(json_file, transform=transform, images=test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Example
    for images, annotations in train_dataloader:
        print(images.shape, annotations)

    # Preprocessing thermal images
    for i in range(len(images)):
        if 'thermal' in annotations[i]:
            images[i] = preprocess_thermal(images[i])

    model = YOLO('yolo11.pt')

    detection_result = detect_objects(images, model)

    # Evaluate detection results
    precision = calculate_precision(detection_result, annotations)
    recall = calculate_recall(detection_result, annotations)
    f1_score = calculate_F1(precision, recall)
    print(f"F1 Score: {f1_score}")

    # Visualize detection results
    display_image_and_annotation(images[0], detection_result[0], segmentation_mask)

if __name__ == "__main__":
    main()
    