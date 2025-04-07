import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(3, 64)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)
        self.enc5 = self.encoder_block(512, 1024)

        # Decoder
        self.dec5 = self.decoder_block(1024, 512)
        self.dec4 = self.decoder_block(512, 256)
        self.dec3 = self.decoder_block(256, 128)
        self.dec2 = self.decoder_block(128, 64)
        self.dec1 = self.decoder_block(64, 64)

        # Final layer
        self.final_layer = nn.Conv2d(64, 1, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Decoder with skip connections
        dec5 = self.dec5(enc5)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec4 = self.dec4(dec5)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec1 = self.dec1(dec2)

        # Final layer
        output = self.final_layer(dec1)

        return output
