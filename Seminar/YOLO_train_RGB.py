from ultralytics import YOLO
print("loading YOLOv11")
# Load the pre-trained YOLOv11 model
model = YOLO('yolov11.pt')

print("loading YOLOv11 done")
# Set the model to training mode
model.trainable = True

"""for name, param in model.model.named_parameters():
    if 'head' not in name:
        param.requries_grad = False"""

print("training model")


# Train the model with your custom dataset and data augmentation
model.train(
    data='/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/data_RGB.yaml',
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,  # Use GPU if available, set to 'cpu' if no GPU
    augment=True,  # Enable data augmentation
    hsv_h=0.015,  # Hue augmentation
    hsv_s=0.7,    # Saturation augmentation
    hsv_v=0.4,    # Value augmentation
    degrees=0.0,  # Rotation degrees
    translate=0.1,  # Translation fraction
    scale=0.5,    # Scaling fraction
    shear=0.0,    # Shear fraction
    perspective=0.0,  # Perspective fraction
    flipud=0.0,   # Vertical flip probability
    fliplr=0.5,   # Horizontal flip probability
    mosaic=1.0,   # Mosaic data augmentation probability
    mixup=0.0     # Mixup data augmentation probability
)

# Save the trained model weights
model_save_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/models/RGB/best.pt'
model.save(model_save_path)

print(f"Model saved to {model_save_path}")
