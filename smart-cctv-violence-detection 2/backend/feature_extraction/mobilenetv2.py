# import torch
# import torch.nn as nn
# from torchvision import models
# import sys
# sys.path.append('..')
# from config import *

# class MobileNetV2FeatureExtractor(nn.Module):
#     def __init__(self, pretrained=True):
#         super(MobileNetV2FeatureExtractor, self).__init__()
        
#         # Load pretrained MobileNetV2
#         self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
#         # Remove the classifier layer
#         self.features = self.mobilenet.features
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # Freeze early layers
#         for param in self.features[:10].parameters():
#             param.requires_grad = False
    
#     def forward(self, x):
#         # x shape: (batch_size, sequence_length, channels, height, width)
#         batch_size, seq_len, c, h, w = x.shape
        
#         # Reshape to process all frames at once
#         x = x.view(batch_size * seq_len, c, h, w)
        
#         # Extract features
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
        
#         # Reshape back to sequence format
#         x = x.view(batch_size, seq_len, -1)
        
#         return x

# def test_feature_extractor():
#     """Test the MobileNetV2 feature extractor"""
#     model = MobileNetV2FeatureExtractor()
#     model.eval()
    
#     # Test input: batch_size=2, sequence_length=16, channels=3, height=224, width=224
#     test_input = torch.randn(2, SEQUENCE_LENGTH, 3, 224, 224)
    
#     with torch.no_grad():
#         features = model(test_input)
    
#     print(f"Input shape: {test_input.shape}")
#     print(f"Output features shape: {features.shape}")
#     print(f"Expected feature dimension: {MOBILENET_FEATURES}")

# if __name__ == "__main__":
#     test_feature_extractor()


import torch
import torch.nn as nn
from torchvision import models
import os
import sys

# Fix import path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config import *

class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, freeze_layers=True):
        super().__init__()

        # Load MobileNetV2 safely (new torchvision API compatible)
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # Feature extractor backbone
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze early layers for stability
        if freeze_layers:
            for param in self.features[:10].parameters():
                param.requires_grad = False

        # Feature dimension safety check
        self.feature_dim = MOBILENET_FEATURES  # should be 1280

    def forward(self, x):
        """
        x shape: [B, T, 3, H, W]
        return: [B, T, 1280]
        """
        b, t, c, h, w = x.shape

        # Merge batch and temporal dimensions
        x = x.view(b * t, c, h, w)

        # CNN forward
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B*T, 1280]

        # Restore temporal dimension
        x = x.view(b, t, self.feature_dim)

        return x


# -------------------------------------------------
# Sanity Test
# -------------------------------------------------
if __name__ == "__main__":
    print("🔍 Testing MobileNetV2FeatureExtractor")

    model = MobileNetV2FeatureExtractor()
    model.eval()

    dummy_input = torch.randn(2, SEQUENCE_LENGTH, 3, 224, 224)

    with torch.no_grad():
        features = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", features.shape)
    print("Expected    :", (2, SEQUENCE_LENGTH, MOBILENET_FEATURES))
