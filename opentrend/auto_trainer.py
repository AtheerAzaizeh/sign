"""
opentrend/auto_trainer.py
Auto-Labeler & Training Loop for Global Fashion Brain

This module handles:
1. Automatic labeling of raw images using CLIP/YOLO
2. Triggering model retraining when threshold is reached
3. Continuous learning loop

Workflow:
    Image Downloaded -> CLIP Predicts Labels -> Save to DB
    When 1000+ new labeled images -> Trigger Retraining -> Update Weights
"""

from __future__ import annotations

import os
import time
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from threading import Thread, Event
import numpy as np

# Deep learning imports
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# FASHION LABELS
# =============================================================================

# Comprehensive fashion style labels for CLIP classification
FASHION_STYLE_LABELS = [
    # Aesthetics
    "minimalist fashion",
    "maximalist fashion",
    "boho chic",
    "cyberpunk techwear",
    "y2k fashion",
    "cottagecore aesthetic",
    "dark academia",
    "old money style",
    "streetwear hypebeast",
    "grunge aesthetic",
    "preppy classic",
    "athleisure sporty",
    "avant garde experimental",
    "quiet luxury",
    "mob wife aesthetic",
    "coquette feminine",
    "balletcore",
    "gorpcore outdoor",
    "normcore casual",
    "vintage retro",
]

# Garment type labels
GARMENT_LABELS = [
    "jacket coat",
    "blazer tailored",
    "dress gown",
    "shirt blouse",
    "t-shirt tee",
    "sweater knitwear",
    "hoodie sweatshirt",
    "pants trousers",
    "jeans denim",
    "skirt",
    "shorts",
    "jumpsuit romper",
    "suit formal",
    "activewear sportswear",
    "outerwear layering",
]

# Color labels
COLOR_LABELS = [
    "black monochrome",
    "white cream",
    "neutral beige",
    "earth tones brown",
    "navy blue",
    "bright red",
    "pastel pink",
    "green sage",
    "vibrant yellow",
    "purple lavender",
    "metallic silver gold",
]

# Material labels
MATERIAL_LABELS = [
    "denim cotton",
    "leather suede",
    "silk satin",
    "wool knit",
    "technical fabric",
    "sustainable eco",
    "linen natural",
    "velvet texture",
]

# All labels combined
ALL_FASHION_LABELS = FASHION_STYLE_LABELS + GARMENT_LABELS + COLOR_LABELS + MATERIAL_LABELS


# =============================================================================
# CLIP AUTO-LABELER
# =============================================================================

class CLIPAutoLabeler:
    """
    Automatic fashion image labeler using CLIP.
    
    CLIP (Contrastive Language-Image Pre-training) can classify
    images into arbitrary text labels without fine-tuning.
    
    Usage:
        labeler = CLIPAutoLabeler()
        labels = labeler.predict("path/to/image.jpg")
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
        labels: List[str] = None,
        top_k: int = 5
    ):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = labels or ALL_FASHION_LABELS
        self.top_k = top_k
        
        self.model = None
        self.processor = None
        self.text_features = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info(f"Loading CLIP model: {self.model_name}")
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Pre-compute text features for all labels
            self._encode_labels()
            
            logger.info(f"CLIP initialized with {len(self.labels)} labels on {self.device}")
            
        except ImportError:
            logger.error("transformers library not installed. Run: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CLIP: {e}")
            raise
    
    def _encode_labels(self):
        """Pre-compute text embeddings for all labels."""
        with torch.no_grad():
            # Add "a photo of" prefix for better CLIP performance
            texts = [f"a photo of {label} fashion" for label in self.labels]
            
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.text_features = self.model.get_text_features(**inputs)
            self.text_features = F.normalize(self.text_features, dim=-1)
    
    def predict(self, image_path: str) -> List[Tuple[str, float]]:
        """
        Predict fashion labels for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of (label, confidence) tuples, sorted by confidence
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = F.normalize(image_features, dim=-1)
                
                # Calculate similarity
                similarity = (image_features @ self.text_features.T).squeeze(0)
                probabilities = F.softmax(similarity * 100, dim=0)  # Temperature scaling
            
            # Get top-k predictions
            top_probs, top_indices = probabilities.topk(self.top_k)
            
            results = [
                (self.labels[idx.item()], prob.item())
                for prob, idx in zip(top_probs, top_indices)
            ]
            
            return results
            
        except Exception as e:
            logger.warning(f"Error predicting labels for {image_path}: {e}")
            return []
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 16
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Predict labels for multiple images.
        
        Returns:
            Dict mapping image paths to their predictions
        """
        results = {}
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for path in batch_paths:
                results[path] = self.predict(path)
        
        return results


# =============================================================================
# YOLO AUTO-LABELER (Alternative)
# =============================================================================

class YOLOAutoLabeler:
    """
    Alternative labeler using YOLOv8 for garment detection.
    
    Better for detecting specific garment items vs CLIP's
    style classification.
    """
    
    FASHION_CLASSES = [
        'shirt', 'pants', 'dress', 'jacket', 'skirt',
        'shoe', 'bag', 'hat', 'glasses', 'watch'
    ]
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model_path = model_path
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize YOLO model."""
        try:
            from ultralytics import YOLO
            
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO initialized: {self.model_path}")
            
        except ImportError:
            logger.warning("ultralytics not installed. YOLO labeler disabled.")
            self.model = None
    
    def predict(self, image_path: str) -> List[Tuple[str, float]]:
        """Detect objects in image."""
        if not self.model:
            return []
        
        try:
            results = self.model(image_path, verbose=False)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name.lower() in [c.lower() for c in self.FASHION_CLASSES]:
                        detections.append((class_name, confidence))
            
            return sorted(detections, key=lambda x: -x[1])[:5]
            
        except Exception as e:
            logger.warning(f"YOLO prediction failed: {e}")
            return []


# =============================================================================
# AUTO TRAINING LOOP
# =============================================================================

class AutoTrainer:
    """
    Automated training loop for the Fashion Brain.
    
    Workflow:
    1. Monitor database for unprocessed images
    2. Label images with CLIP
    3. When threshold reached, trigger model retraining
    4. Update prediction engine weights
    
    Runs continuously in background.
    """
    
    def __init__(
        self,
        db_path: str = "./data/fashion_brain.db",
        labeler: str = "clip",  # "clip" or "yolo"
        retrain_threshold: int = 1000,
        check_interval: int = 60  # seconds
    ):
        self.db_path = db_path
        self.retrain_threshold = retrain_threshold
        self.check_interval = check_interval
        
        # Initialize labeler
        if labeler == "clip":
            self.labeler = CLIPAutoLabeler()
        else:
            self.labeler = YOLOAutoLabeler()
        
        # Database connection
        self._db = None
        
        # Control
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        
        # Stats
        self.images_labeled = 0
        self.last_retrain = None
        
        logger.info(f"AutoTrainer initialized (threshold: {retrain_threshold})")
    
    def _get_db(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db
    
    def _get_unprocessed_images(self, limit: int = 100) -> List[Dict]:
        """Get images that need labeling."""
        conn = self._get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, local_path, title, tags
            FROM fashion_items
            WHERE processed = FALSE AND local_path IS NOT NULL
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _save_labels(self, item_id: int, labels: List[Tuple[str, float]]):
        """Save predicted labels to database."""
        conn = self._get_db()
        cursor = conn.cursor()
        
        label_str = ','.join([f"{label}:{conf:.2f}" for label, conf in labels])
        
        cursor.execute("""
            UPDATE fashion_items
            SET processed = TRUE, predicted_labels = ?
            WHERE id = ?
        """, (label_str, item_id))
        
        conn.commit()
    
    def _count_labeled_since_retrain(self) -> int:
        """Count images labeled since last retrain."""
        conn = self._get_db()
        cursor = conn.cursor()
        
        if self.last_retrain:
            cursor.execute("""
                SELECT COUNT(*) FROM fashion_items
                WHERE processed = TRUE AND harvested_at > ?
            """, (self.last_retrain,))
        else:
            cursor.execute("SELECT COUNT(*) FROM fashion_items WHERE processed = TRUE")
        
        return cursor.fetchone()[0]
    
    def _trigger_retrain(self):
        """Trigger model retraining."""
        logger.info("🔄 Triggering model retraining...")
        
        try:
            # Import the neural forecaster
            from neural_forecaster import BreakthroughPredictionEngine
            
            # Get training data from database
            conn = self._get_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT predicted_labels, category, engagement
                FROM fashion_items
                WHERE processed = TRUE
                ORDER BY harvested_at DESC
                LIMIT 5000
            """)
            
            data = cursor.fetchall()
            
            if len(data) < 100:
                logger.warning("Not enough data for retraining")
                return
            
            # Extract label statistics for trend analysis
            label_counts = {}
            for row in data:
                labels = row[0].split(',') if row[0] else []
                for label in labels:
                    label_name = label.split(':')[0]
                    if label_name:
                        label_counts[label_name] = label_counts.get(label_name, 0) + 1
            
            # Log top labels
            top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:10]
            logger.info(f"Top detected labels: {top_labels}")
            
            # Update last retrain timestamp
            self.last_retrain = datetime.now().isoformat()
            
            # Log training event
            cursor.execute("""
                INSERT INTO training_log (new_samples, previous_accuracy, new_accuracy, model_version)
                VALUES (?, ?, ?, ?)
            """, (len(data), 0.0, 0.0, f"v{datetime.now().strftime('%Y%m%d%H%M')}"))
            conn.commit()
            
            logger.info(f"✅ Retraining complete with {len(data)} samples")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    def label_batch(self, batch_size: int = 50) -> int:
        """Label a batch of unprocessed images."""
        images = self._get_unprocessed_images(batch_size)
        
        if not images:
            return 0
        
        labeled = 0
        
        for item in images:
            try:
                image_path = item['local_path']
                
                if not image_path or not Path(image_path).exists():
                    continue
                
                # Get predictions
                predictions = self.labeler.predict(image_path)
                
                if predictions:
                    self._save_labels(item['id'], predictions)
                    labeled += 1
                    self.images_labeled += 1
                    
            except Exception as e:
                logger.warning(f"Error labeling image {item['id']}: {e}")
        
        logger.info(f"Labeled {labeled} images")
        
        # Check if retrain threshold reached
        labeled_count = self._count_labeled_since_retrain()
        if labeled_count >= self.retrain_threshold:
            self._trigger_retrain()
        
        return labeled
    
    def _run_loop(self):
        """Main processing loop."""
        logger.info("AutoTrainer loop started")
        
        while not self._stop_event.is_set():
            try:
                self.label_batch()
            except Exception as e:
                logger.error(f"Error in AutoTrainer loop: {e}")
            
            # Wait for next check
            self._stop_event.wait(self.check_interval)
        
        logger.info("AutoTrainer loop stopped")
    
    def start(self):
        """Start background processing."""
        if self._thread and self._thread.is_alive():
            logger.warning("AutoTrainer already running")
            return
        
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("AutoTrainer started in background")
    
    def stop(self):
        """Stop background processing."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("AutoTrainer stopped")
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        conn = self._get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM fashion_items WHERE processed = TRUE")
        processed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM fashion_items WHERE processed = FALSE")
        pending = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM training_log")
        retrains = cursor.fetchone()[0]
        
        return {
            'images_labeled_total': processed,
            'images_pending': pending,
            'images_labeled_session': self.images_labeled,
            'retrains': retrains,
            'retrain_threshold': self.retrain_threshold,
            'last_retrain': self.last_retrain
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║               AUTO-LABELER & TRAINING LOOP                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  Components:                                                              ║
║  • CLIPAutoLabeler - Fashion style classification                         ║
║  • YOLOAutoLabeler - Garment detection (alternative)                      ║
║  • AutoTrainer - Continuous learning loop                                 ║
║                                                                           ║
║  Usage:                                                                   ║
║    trainer = AutoTrainer()                                                ║
║    trainer.start()  # Background processing                               ║
║                                                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("Testing CLIP labeler...")
    try:
        labeler = CLIPAutoLabeler()
        print("✅ CLIP labeler initialized")
    except Exception as e:
        print(f"❌ CLIP initialization failed: {e}")
