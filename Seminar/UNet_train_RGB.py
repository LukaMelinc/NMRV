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
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def show_augmented_images(dataloader, num_samples=3):
    batch = next(iter(dataloader))  # Get one batch
    images, masks  = batch

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

    for i in range(num_samples):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)  # Convert back to 0-255 for display

        mask = masks[i].squeeze()

        axes[i, 0].imshow(img)  # Show image
        axes[i, 0].set_title("Augmentirana slika")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")  # Show mask
        axes[i, 1].set_title("Augmentirana maska")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

#from your_module import UNet, SegmentationDataset  # Make sure to import your UNet model and SegmentationDataset
# Define the SegmentationDataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask_np = np.array(mask)
        print(f"Unique values in mask before adaptation: {np.unique(mask_np)}")


        # Adapt the mask to ignore classes 6 and 7, and consider class 5 as part of class 3
        mask = torch.from_numpy(np.array(mask))
        mask[mask == 5] = 3
        mask[mask == 6] = 0
        mask[mask == 7] = 0

        return image, mask
# Define the transformations for the training and validation sets
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transformations if needed
])

# Create the training and validation datasets
train_dataset = SegmentationDataset(
    image_dir='/content/drive/MyDrive/RGB_datasets_segmentation_V2/images/train',
    mask_dir='/content/drive/MyDrive/RGB_datasets_segmentation_V2/masks/train',
    transform=transform
)

val_dataset = SegmentationDataset(
    image_dir='/content/drive/MyDrive/RGB_datasets_segmentation_V2/images/val',
    mask_dir='/content/drive/MyDrive/RGB_datasets_segmentation_V2/masks/val',
    transform=transform
)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=1)

# Initialize the UNet model
model = UNet(num_classes=4)  # Adjust the number of classes as needed

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        show_augmented_images(train_loader)
        images = images.to(device)
        masks = masks.to(device)
        masks = masks.squeeze(1).long()

        print(f"unique values in masks: {torch.unique(masks)}")
        print(f"Masks shape: {masks.shape}")

        # Forward pass
        outputs = model(images)
        print(f"outputs shape: {outputs.shape}")
        print(f"masks shape: {outputs[0, :5, 0, 0]}")
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}')

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.squeeze(1).long()

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss/len(val_loader)}')

print('Training complete')
