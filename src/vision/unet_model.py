"""
U-Net Segmentation Model (Advanced Mode)
-----------------------------------------
Deep learning segmentation architecture for medical image analysis.
NOTE: This is a scaffold. For production use, train with NIH Chest X-ray
or CheXpert datasets.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation.

    Architecture:
        Encoder: 4 downsampling blocks (64→128→256→512)
        Bottleneck: 1024 channels
        Decoder: 4 upsampling blocks with skip connections
        Output: 1-channel segmentation mask
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoder path with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


# ────────────────────────────────────────────────────────────────
# Prediction utilities (requires trained weights)
# ────────────────────────────────────────────────────────────────

MODEL_WEIGHTS_PATH = None  # Set to path of trained weights


def predict_segmentation(image_path, weights_path=None):
    """
    Generate a segmentation mask for a medical image using U-Net.

    NOTE: Requires trained model weights. Without weights, returns
    a placeholder message.

    To train:
        1. Download NIH Chest X-ray dataset or CheXpert dataset
        2. Prepare binary masks for abnormal regions
        3. Train UNet with BCE + Dice loss
        4. Save weights and set MODEL_WEIGHTS_PATH

    Args:
        image_path: Path to the input medical image
        weights_path: Path to trained model weights (.pth)

    Returns:
        dict with keys: available, mask, overlay, message
    """
    wp = weights_path or MODEL_WEIGHTS_PATH

    if wp is None:
        return {
            "available": False,
            "mask": None,
            "overlay": None,
            "message": (
                "U-Net model weights not found. "
                "For accurate detection, train a U-Net model using "
                "NIH Chest X-ray or CheXpert datasets. "
                "The architecture is ready — only trained weights are needed."
            )
        }

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(torch.load(wp, map_location=device))
        model.eval()

        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        with torch.no_grad():
            mask = model(tensor).squeeze().cpu().numpy()

        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # Create colored overlay
        original = cv2.imread(image_path)
        original = cv2.resize(original, (256, 256))
        overlay = original.copy()
        colored_mask = np.zeros_like(original)
        colored_mask[:, :, 2] = mask_binary  # Red channel
        cv2.addWeighted(colored_mask, 0.4, overlay, 0.6, 0, overlay)

        return {
            "available": True,
            "mask": mask_binary,
            "overlay": overlay,
            "message": "Segmentation completed successfully."
        }

    except Exception as e:
        return {
            "available": False,
            "mask": None,
            "overlay": None,
            "message": f"U-Net prediction failed: {str(e)}"
        }
