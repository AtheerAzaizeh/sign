"""
opentrend/vision/feature_extractor.py
ResNet50 Feature Vector Extraction for Fashion Items
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import List, Union


class FashionFeatureExtractor:
    """
    Extracts 2048-dimensional feature vectors from garment images
    using a pre-trained ResNet50 backbone.
    
    Why ResNet50?
    - Proven architecture with excellent transfer learning properties
    - Captures hierarchical features: edges → textures → patterns → objects
    - 2048-dim output is rich enough for clustering while manageable
    - Pre-trained on ImageNet generalizes well to fashion imagery
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the feature extractor.
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final classification layer to get feature vectors
        # Original: [..., AdaptiveAvgPool2d, Linear(2048, 1000)]
        # Modified: [..., AdaptiveAvgPool2d] → outputs (batch, 2048, 1, 1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)
        
        # ImageNet normalization (required for pre-trained weights)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ])
    
    def extract(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Extract feature vector from a single image.
        
        Args:
            image: File path or PIL Image
            
        Returns:
            numpy array of shape (2048,) - the feature vector
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be file path or PIL Image")
        
        # Transform and add batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features (no gradient computation needed)
        with torch.no_grad():
            features = self.model(tensor)
        
        # Flatten from (1, 2048, 1, 1) to (2048,)
        feature_vector = features.squeeze().cpu().numpy()
        
        return feature_vector
    
    def extract_batch(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Extract feature vectors from multiple images efficiently.
        
        Args:
            images: List of file paths or PIL Images
            
        Returns:
            numpy array of shape (N, 2048) where N = len(images)
        """
        tensors = []
        
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            tensors.append(self.transform(img))
        
        # Stack into batch tensor
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
        
        # Shape: (N, 2048)
        return features.squeeze(-1).squeeze(-1).cpu().numpy()


# Usage Example
if __name__ == "__main__":
    # Initialize extractor
    extractor = FashionFeatureExtractor()
    
    # Example 1: Single image
    feature_vector = extractor.extract("silver_jacket.jpg")
    print(f"Feature vector shape: {feature_vector.shape}")  # (2048,)
    print(f"Feature vector sample: {feature_vector[:5]}")
    
    # Example 2: Batch processing
    image_paths = [
        "dress_001.jpg",
        "jacket_002.jpg",
        "pants_003.jpg"
    ]
    batch_features = extractor.extract_batch(image_paths)
    print(f"Batch features shape: {batch_features.shape}")  # (3, 2048)
