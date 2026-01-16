"""
opentrend/ai_processor.py
AI Filter & Auto-Labeler for Global Fashion Brain

This module provides:
- CLIP-based fashion classification
- "Garbage Filter" to discard non-fashion images (confidence < 0.85)
- Style auto-tagging for valid images
- Batch processing for efficiency

Key Feature: The Garbage Filter
    If an image's probability of being "Fashion/Clothing" is < 0.85,
    it is IMMEDIATELY DISCARDED to prevent database pollution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# FASHION LABEL TAXONOMY
# =============================================================================

# Primary validation labels (for garbage filter)
FASHION_VALIDATION_LABELS = [
    "a photo of clothing or fashion",
    "a photo of a person wearing clothes",
    "a photo of a fashion outfit",
    "a photo of a garment or apparel",
    "a photo of accessories or jewelry",
]

NON_FASHION_LABELS = [
    "a photo of food or meal",
    "a photo of a meme or joke",
    "a photo of text or document",
    "a photo of landscape or nature",
    "a photo of animals or pets",
    "a photo of a screenshot",
    "a photo of art or illustration",
]

# Style classification labels
STYLE_LABELS = [
    "streetwear urban style",
    "y2k retro fashion",
    "minimalist clean style",
    "bohemian boho style",
    "cyberpunk techwear",
    "vintage retro fashion",
    "preppy classic style",
    "grunge alternative style",
    "athleisure sporty style",
    "quiet luxury elegant",
    "avant garde experimental",
    "cottagecore romantic",
    "dark academia style",
    "coquette feminine style",
]

# Garment type labels
GARMENT_LABELS = [
    "jacket or coat",
    "dress or gown",
    "shirt or blouse",
    "t-shirt or tee",
    "sweater or knitwear",
    "pants or trousers",
    "jeans or denim",
    "skirt",
    "shorts",
    "suit or blazer",
    "activewear or sportswear",
    "accessories",
]

# Color labels
COLOR_LABELS = [
    "black or dark colors",
    "white or light colors",
    "neutral beige tones",
    "earth tones brown",
    "blue shades",
    "red or pink",
    "green shades",
    "yellow or gold",
    "purple or lavender",
    "metallic or shiny",
]


# =============================================================================
# FASHION AI PROCESSOR
# =============================================================================

@dataclass
class ProcessingResult:
    """Result of AI processing."""
    is_valid: bool
    fashion_confidence: float
    predicted_styles: List[Tuple[str, float]]
    predicted_garments: List[Tuple[str, float]]
    predicted_colors: List[Tuple[str, float]]
    all_labels: List[str]


class FashionAIProcessor:
    """
    AI processor for fashion image classification and filtering.
    
    Uses CLIP (Contrastive Language-Image Pre-training) for:
    1. Garbage Filter: Reject non-fashion images (< 85% confidence)
    2. Style Classification: Predict fashion style tags
    3. Garment Detection: Identify garment types
    4. Color Analysis: Detect dominant colors
    
    Usage:
        processor = FashionAIProcessor()
        
        # Quick validation
        is_valid, labels, confidence = processor.process("image.jpg")
        
        # Full analysis
        result = processor.analyze("image.jpg")
    """
    
    # Confidence threshold for garbage filter
    FASHION_THRESHOLD = 0.85
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
        fashion_threshold: float = 0.85
    ):
        """
        Initialize the AI processor.
        
        Args:
            model_name: HuggingFace CLIP model name
            device: torch device (cuda/cpu)
            fashion_threshold: Minimum confidence for fashion images
        """
        self.model_name = model_name
        self.device = torch.device(device) if device else DEVICE
        self.fashion_threshold = fashion_threshold
        
        self.model = None
        self.processor = None
        
        # Pre-computed text features
        self._fashion_features = None
        self._non_fashion_features = None
        self._style_features = None
        self._garment_features = None
        self._color_features = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize CLIP model and pre-compute embeddings."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info(f"Loading CLIP model: {self.model_name}")
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Pre-compute all text embeddings
            self._precompute_embeddings()
            
            logger.info(f"CLIP initialized on {self.device}")
            
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"CLIP initialization failed: {e}")
            raise
    
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode text labels to embeddings."""
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_text_features(**inputs)
            return F.normalize(features, dim=-1)
    
    def _precompute_embeddings(self):
        """Pre-compute text embeddings for all label categories."""
        logger.info("Pre-computing text embeddings...")
        
        self._fashion_features = self._encode_texts(FASHION_VALIDATION_LABELS)
        self._non_fashion_features = self._encode_texts(NON_FASHION_LABELS)
        self._style_features = self._encode_texts(
            [f"a photo of {s} fashion" for s in STYLE_LABELS]
        )
        self._garment_features = self._encode_texts(
            [f"a photo of {g}" for g in GARMENT_LABELS]
        )
        self._color_features = self._encode_texts(
            [f"clothing in {c}" for c in COLOR_LABELS]
        )
        
        logger.info("Text embeddings ready")
    
    def _get_image_features(self, image_path: str) -> Optional[torch.Tensor]:
        """Get image features from CLIP."""
        try:
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                return F.normalize(features, dim=-1)
                
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
            return None
    
    def _compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between image and text features."""
        return (image_features @ text_features.T).squeeze(0)
    
    def is_fashion(self, image_path: str) -> Tuple[bool, float]:
        """
        Check if image contains fashion content.
        
        This is the GARBAGE FILTER. Images with confidence < threshold
        are rejected to prevent database pollution.
        
        Args:
            image_path: Path to image file
            
        Returns:
            (is_valid, confidence_score)
        """
        image_features = self._get_image_features(image_path)
        if image_features is None:
            return False, 0.0
        
        # Calculate fashion vs non-fashion scores
        fashion_sim = self._compute_similarity(image_features, self._fashion_features)
        non_fashion_sim = self._compute_similarity(image_features, self._non_fashion_features)
        
        # Average scores
        fashion_score = fashion_sim.mean().item()
        non_fashion_score = non_fashion_sim.mean().item()
        
        # Normalize to probability
        total = np.exp(fashion_score * 10) + np.exp(non_fashion_score * 10)
        fashion_prob = np.exp(fashion_score * 10) / total
        
        is_valid = fashion_prob >= self.fashion_threshold
        
        if not is_valid:
            logger.debug(f"Garbage filtered: {image_path} (conf={fashion_prob:.2%})")
        
        return is_valid, fashion_prob
    
    def classify_style(
        self,
        image_features: torch.Tensor,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Classify fashion style."""
        similarities = self._compute_similarity(image_features, self._style_features)
        probs = F.softmax(similarities * 10, dim=0)
        
        top_probs, top_indices = probs.topk(top_k)
        
        return [
            (STYLE_LABELS[idx.item()], prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]
    
    def classify_garment(
        self,
        image_features: torch.Tensor,
        top_k: int = 2
    ) -> List[Tuple[str, float]]:
        """Classify garment type."""
        similarities = self._compute_similarity(image_features, self._garment_features)
        probs = F.softmax(similarities * 10, dim=0)
        
        top_probs, top_indices = probs.topk(top_k)
        
        return [
            (GARMENT_LABELS[idx.item()], prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]
    
    def classify_color(
        self,
        image_features: torch.Tensor,
        top_k: int = 2
    ) -> List[Tuple[str, float]]:
        """Classify dominant colors."""
        similarities = self._compute_similarity(image_features, self._color_features)
        probs = F.softmax(similarities * 10, dim=0)
        
        top_probs, top_indices = probs.topk(top_k)
        
        return [
            (COLOR_LABELS[idx.item()], prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]
    
    def analyze(self, image_path: str) -> Optional[ProcessingResult]:
        """
        Full analysis of a fashion image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ProcessingResult with all classifications, or None if invalid
        """
        # First: Garbage filter
        is_valid, fashion_conf = self.is_fashion(image_path)
        
        if not is_valid:
            return ProcessingResult(
                is_valid=False,
                fashion_confidence=fashion_conf,
                predicted_styles=[],
                predicted_garments=[],
                predicted_colors=[],
                all_labels=[]
            )
        
        # Get image features (recompute for classifications)
        image_features = self._get_image_features(image_path)
        if image_features is None:
            return None
        
        # Classify all categories
        styles = self.classify_style(image_features)
        garments = self.classify_garment(image_features)
        colors = self.classify_color(image_features)
        
        # Combine top labels
        all_labels = []
        for label, conf in styles[:2]:
            if conf > 0.1:
                all_labels.append(label.split()[0])  # First word
        for label, conf in garments[:1]:
            if conf > 0.1:
                all_labels.append(label.split()[0])
        for label, conf in colors[:1]:
            if conf > 0.1:
                all_labels.append(label.split()[0])
        
        return ProcessingResult(
            is_valid=True,
            fashion_confidence=fashion_conf,
            predicted_styles=styles,
            predicted_garments=garments,
            predicted_colors=colors,
            all_labels=all_labels
        )
    
    def process(self, image_path: str) -> Tuple[bool, List[str], float]:
        """
        Quick process method for harvester integration.
        
        Args:
            image_path: Path to image
            
        Returns:
            (is_valid, labels, confidence)
        """
        result = self.analyze(image_path)
        
        if result is None:
            return False, [], 0.0
        
        return result.is_valid, result.all_labels, result.fashion_confidence
    
    def process_batch(
        self,
        image_paths: List[str],
        batch_size: int = 16
    ) -> Dict[str, ProcessingResult]:
        """
        Process multiple images efficiently.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            Dict mapping paths to results
        """
        results = {}
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            
            for path in batch:
                result = self.analyze(path)
                if result:
                    results[path] = result
        
        return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           FASHION AI PROCESSOR                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Features:                                                        ║
║  • Garbage Filter (85% confidence threshold)                      ║
║  • Style Classification (14 fashion styles)                       ║
║  • Garment Detection (12 garment types)                           ║
║  • Color Analysis (10 color categories)                           ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Initializing AI processor...")
    try:
        processor = FashionAIProcessor()
        print("✅ AI Processor ready")
        print(f"   Device: {processor.device}")
        print(f"   Threshold: {processor.fashion_threshold:.0%}")
    except Exception as e:
        print(f"❌ Failed: {e}")
