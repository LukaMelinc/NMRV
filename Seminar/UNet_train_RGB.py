import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import segmentation_models_pytorch as smp
from network import UNet
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.png', '.png'))
        image = np.array(image.open(img_path).convert("RGB"))
        mask = np.array(image.open(mask_path).convert("L"), dtype=np.uint8)


        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
# Data augmentations
transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=35, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()
])



train_dataset = SegmentationDataset(
    image_dir='/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/RGB_images/train/images',
    mask_dir='/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/RGB_images/train/train_annotations',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

val_dataset = SegmentationDataset(
    image_dir='/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/RGB_images/test/images',
    mask_dir='/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/RGB_images/test/test_annotations',
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


model = UNet()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)



num_epochs = 30
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')

    # Save the model if the validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'UNet_best.pth')
        print(f'Model saved as UNet_best.pth')
