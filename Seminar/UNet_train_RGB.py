import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics import JaccardIndex
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set this for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()

        # Encoder
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) # konvolucija, jedro 3x3, 1 piksel za robove
        self.relu1_1 = nn.ReLU(inplace=True) # ReLu nelinearnost
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # manjšanje dimenzije, 2x2 jedro, manjša dimenzijo za 2

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (latentni prostor)
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Obratna konvolucija (parametri so isti)
        self.conv4_3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_4 = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_4 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.conv2_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_4 = nn.ReLU(inplace=True)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_4 = nn.ReLU(inplace=True)

        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1) # 1 out za masko

    # Propagacija skozi mrežo
    def forward(self, x):

        # Encoder
        e1 = self.relu1_1(self.conv1_1(x))
        e1 = self.relu1_2(self.conv1_2(e1))
        p1 = self.pool1(e1)

        e2 = self.relu2_1(self.conv2_1(p1))
        e2 = self.relu2_2(self.conv2_2(e2))
        p2 = self.pool2(e2)

        e3 = self.relu3_1(self.conv3_1(p2))
        e3 = self.relu3_2(self.conv3_2(e3))
        p3 = self.pool3(e3)

        e4 = self.relu4_1(self.conv4_1(p3))
        e4 = self.relu4_2(self.conv4_2(e4))
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck_relu1(self.bottleneck_conv1(p4))
        b = self.bottleneck_relu2(self.bottleneck_conv2(b))

        # Decoder
        d4 = self.upconv4(b)
        #d4 = torch.cat((d4, e4), dim=1) # direktna povezava (siva puščica)
        diffY = e4.size()[2] - d4.size()[2]
        diffX = e4.size()[3] - d4.size()[3]
        d4 = F.pad(d4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.relu4_3(self.conv4_3(d4))
        d4 = self.relu4_4(self.conv4_4(d4))

        d3 = self.upconv3(d4)
        #d3 = torch.cat((d3, e3), dim=1) # direktna povezava (siva puščica)
        diffY = e3.size()[2] - d3.size()[2]
        diffX = e3.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.relu3_3(self.conv3_3(d3))
        d3 = self.relu3_4(self.conv3_4(d3))

        d2 = self.upconv2(d3)
        #d2 = torch.cat((d2, e2), dim=1) # direktna povezava (siva puščica)
        diffY = e2.size()[2] - d2.size()[2]
        diffX = e2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.relu2_3(self.conv2_3(d2))
        d2 = self.relu2_4(self.conv2_4(d2))

        d1 = self.upconv1(d2)
        #d1 = torch.cat((d1, e1), dim=1) # direktna povezava (siva puščica)
        diffY = e1.size()[2] - d1.size()[2]
        diffX = e1.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.relu1_3(self.conv1_3(d1))
        d1 = self.relu1_4(self.conv1_4(d1))

        # Final layer
        out = self.final_conv(d1)
        return out
    

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
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
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace('.png', '.png'))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        # Modify the mask to ignore classes 6 and 7, and treat class 5 as class 3
        mask = np.where(mask == 5, 3, mask)  # Treat class 5 as class 3
        mask = np.where(mask >= 6, 0, mask)   # Ignore classes 6 and 7 by setting them to background (class 0)

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
    image_dir='/content/drive/MyDrive/RGB_datasets_segmentation/images/train',
    mask_dir='/content/drive/MyDrive/RGB_datasets_segmentation/masks/train',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

val_dataset = SegmentationDataset(
    image_dir='/content/drive/MyDrive/RGB_datasets_segmentation/images/val',
    mask_dir='/content/drive/MyDrive/RGB_datasets_segmentation/masks/val',
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

model = UNet(in_channels=3, out_channels=4)  # Assuming 4 classes after modifications

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler
model.to(device)

num_epochs = 30
best_val_loss = float('inf')

# Checkpointing
checkpoint_path = 'best_model.pth'

# Evaluation metric
iou_metric = JaccardIndex(task="multiclass", num_classes=4).to(device)

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename='training.log', filemode='w', format='%(message)s')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).long()  # Convert masks to LongTensor

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Validation
    model.eval()
    val_loss = 0
    val_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()  # Convert masks to LongTensor

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            # Compute IoU
            preds = torch.argmax(outputs, dim=1)
            val_iou += iou_metric(preds, masks).item()

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}')
    logging.info(f'Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}')

    # Step the scheduler
    scheduler.step()

    # Save the model if the validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, 'UNet_best.pth')
        print(f'Model saved as UNet_best.pth')
        logging.info(f'Model saved as UNet_best.pth')

    # Early stopping (optional)
    # if val_loss > best_val_loss:
    #     early_stopping_counter += 1
    #     if early_stopping_counter >= patience:
    #         print('Early stopping triggered')
    #         break
    # else:
    #     early_stopping_counter = 0
# Data augmentations for testing (no random transformations)
test_transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()
])

test_dataset = SegmentationDataset(
    image_dir='/content/drive/MyDrive/RGB_images/test',
    mask_dir='/content/drive/MyDrive/RGB_semantic_annotations/masks/test',
    transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

def postprocess_predictions(output):
    """Convert model output to final mask"""
    pred = output.argmax(dim=1)  # Get class indices
    # Any pixels predicted as class 4 should be mapped elsewhere or ignored
    pred[pred == 4] = 3  # Or handle differently based on your needs
    return pred

# Load the trained model for testing
model.load_state_dict(torch.load('/content/drive/MyDrive/models/model.pth'))
model.to(device)
model.eval()

# Testing the model
test_iou = 0
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        iou = iou_metric(outputs.argmax(dim=1), masks.long())
        test_iou += iou.item()

test_iou /= len(test_loader)
print(f'Test IoU: {test_iou:.4f}')

# Visualize some results
def visualize_results(images, masks, outputs, num_images=4):
    fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    for i in range(num_images):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        mask = masks[i].cpu().numpy()
        output = outputs[i].argmax(dim=0).cpu().numpy()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(output, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize the first few test results
images, masks = next(iter(test_loader))
images = images.to(device)
masks = masks.to(device)
outputs = model(images)
visualize_results(images, masks, outputs)