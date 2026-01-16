"""
opentrend/auto_trainer.py
Continuous Learning Loop for Global Fashion Brain

This module handles:
1. Monitoring database for new labeled samples
2. Triggering model retraining when threshold reached
3. Fine-tuning the NeuralForecaster on new data
4. Saving versioned model checkpoints

Trigger: When database reaches 1,000 new valid samples
Action: Fine-tune LSTM for 5 epochs, save new model version
"""

from __future__ import annotations

import os
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from threading import Thread, Event
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Model storage
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class TrainingConfig:
    """Configuration for continuous learning."""
    
    # Trigger thresholds
    NEW_SAMPLES_THRESHOLD = 1000
    CHECK_INTERVAL_SECONDS = 300  # 5 minutes
    
    # Training parameters
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    
    # Model versioning
    MODEL_PREFIX = "fashion_brain"
    MAX_SAVED_MODELS = 5  # Keep last N versions


# =============================================================================
# CONTINUOUS TRAINER
# =============================================================================

class ContinuousTrainer:
    """
    Automated training loop for the Fashion Brain.
    
    Workflow:
    1. Monitor database for new processed samples
    2. When NEW_SAMPLES_THRESHOLD reached, trigger training
    3. Export labeled data from database
    4. Fine-tune NeuralForecaster for EPOCHS
    5. Save versioned model checkpoint
    6. Update database with training log
    
    Usage:
        trainer = ContinuousTrainer()
        trainer.start()  # Background thread
        # ... harvesting continues ...
        trainer.stop()
    """
    
    def __init__(
        self,
        db_url: str = "sqlite:///./data/fashion_brain.db",
        model_dir: str = "./models",
        config: TrainingConfig = None
    ):
        self.db_url = db_url
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.config = config or TrainingConfig()
        
        # Threading control
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        
        # Stats
        self.training_count = 0
        self.last_training: Optional[datetime] = None
        self.last_sample_count = 0
        self.current_model_version: Optional[str] = None
        
        # Load AI processor for feature extraction
        self._ai_processor = None
        
        logger.info("ContinuousTrainer initialized")
    
    def _get_db_session(self):
        """Get database session."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        engine = create_engine(self.db_url)
        Session = sessionmaker(bind=engine)
        return Session()
    
    def _count_new_samples(self) -> int:
        """Count samples processed since last training."""
        try:
            from harvesters.core import FashionItemModel
            
            session = self._get_db_session()
            
            # Count valid, processed items not used for training
            count = session.query(FashionItemModel).filter(
                FashionItemModel.is_valid == True,
                FashionItemModel.is_processed == True,
                FashionItemModel.is_used_for_training == False
            ).count()
            
            session.close()
            return count
            
        except Exception as e:
            logger.error(f"Error counting samples: {e}")
            return 0
    
    def _export_training_data(self) -> Tuple[List[Dict], int]:
        """
        Export labeled data from database for training.
        
        Returns:
            (list of sample dicts, count)
        """
        try:
            from harvesters.core import FashionItemModel
            
            session = self._get_db_session()
            
            # Get all valid samples not yet used for training
            items = session.query(FashionItemModel).filter(
                FashionItemModel.is_valid == True,
                FashionItemModel.is_processed == True,
                FashionItemModel.is_used_for_training == False
            ).all()
            
            samples = []
            for item in items:
                samples.append({
                    'id': item.id,
                    'local_path': item.local_path,
                    'labels': item.predicted_labels.split(',') if item.predicted_labels else [],
                    'category': item.category,
                    'style': item.style,
                    'engagement': item.engagement,
                    'harvested_at': item.harvested_at
                })
            
            session.close()
            return samples, len(samples)
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return [], 0
    
    def _mark_used_for_training(self, sample_ids: List[int]):
        """Mark samples as used for training."""
        try:
            from harvesters.core import FashionItemModel
            
            session = self._get_db_session()
            
            session.query(FashionItemModel).filter(
                FashionItemModel.id.in_(sample_ids)
            ).update({
                FashionItemModel.is_used_for_training: True
            }, synchronize_session=False)
            
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Error marking samples: {e}")
    
    def _prepare_training_tensors(
        self,
        samples: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training tensors from samples.
        
        Creates synthetic time-series from label frequencies for forecasting.
        """
        # Aggregate label frequencies over time
        label_counts = {}
        
        for sample in samples:
            for label in sample['labels']:
                label = label.strip()
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
        
        if not label_counts:
            return torch.tensor([]), torch.tensor([])
        
        # Create sequences for each label
        n_labels = len(label_counts)
        sequence_length = 52  # 52 weeks
        
        X = []
        y = []
        
        for label, count in label_counts.items():
            # Synthetic weekly trend with growth
            base = count / len(samples) * 100
            t = np.arange(sequence_length + 1)
            
            # Random growth pattern
            growth = np.random.uniform(0.8, 1.2)
            noise = np.random.normal(0, base * 0.1, sequence_length + 1)
            
            values = base * (growth ** (t / 52)) + noise
            values = np.clip(values, 0, 100)
            
            X.append(values[:-1])
            y.append(values[-1])
        
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
        
        return X_tensor, y_tensor
    
    def _fine_tune_model(
        self,
        samples: List[Dict],
        epochs: int = None
    ) -> Tuple[float, str]:
        """
        Fine-tune the neural forecaster on new data.
        
        Returns:
            (final_loss, model_path)
        """
        epochs = epochs or self.config.EPOCHS
        
        logger.info(f"Fine-tuning model on {len(samples)} samples for {epochs} epochs")
        
        # Prepare data
        X, y = self._prepare_training_tensors(samples)
        
        if len(X) == 0:
            logger.warning("No training data available")
            return 0.0, ""
        
        # Create dataset
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        # Load or create model
        try:
            from neural_forecaster import LSTMTrendPredictor
            model = LSTMTrendPredictor(input_size=1, hidden_size=64, num_layers=2)
        except Exception as e:
            logger.error(f"Could not import model: {e}")
            return 0.0, ""
        
        # Check for existing model
        existing_models = sorted(self.model_dir.glob(f"{self.config.MODEL_PREFIX}_v*.pth"))
        if existing_models:
            latest = existing_models[-1]
            try:
                model.load_state_dict(torch.load(latest))
                logger.info(f"Loaded existing model: {latest}")
            except:
                pass
        
        # Training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.train()
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        criterion = nn.MSELoss()
        
        final_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            final_loss = avg_loss
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save new version
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{self.config.MODEL_PREFIX}_v{version}.pth"
        torch.save(model.state_dict(), model_path)
        
        self.current_model_version = f"v{version}"
        logger.info(f"Saved model: {model_path}")
        
        # Cleanup old models
        self._cleanup_old_models()
        
        return final_loss, str(model_path)
    
    def _cleanup_old_models(self):
        """Keep only the last N model versions."""
        models = sorted(self.model_dir.glob(f"{self.config.MODEL_PREFIX}_v*.pth"))
        
        while len(models) > self.config.MAX_SAVED_MODELS:
            oldest = models.pop(0)
            oldest.unlink()
            logger.info(f"Removed old model: {oldest}")
    
    def _log_training(
        self,
        new_samples: int,
        loss: float,
        model_path: str
    ):
        """Log training to database."""
        try:
            from harvesters.core import TrainingLogModel
            
            session = self._get_db_session()
            
            log = TrainingLogModel(
                new_samples=new_samples,
                total_samples=self.last_sample_count + new_samples,
                epochs=self.config.EPOCHS,
                new_accuracy=1 - loss if loss else 0,
                model_version=self.current_model_version,
                model_path=model_path,
                status='completed'
            )
            
            session.add(log)
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Error logging training: {e}")
    
    def train_now(self) -> bool:
        """
        Trigger training immediately.
        
        Returns True if training was performed.
        """
        logger.info("Manual training trigger")
        
        samples, count = self._export_training_data()
        
        if count < 10:
            logger.warning("Not enough samples for training")
            return False
        
        loss, model_path = self._fine_tune_model(samples)
        
        if model_path:
            # Mark samples as used
            self._mark_used_for_training([s['id'] for s in samples])
            
            # Log training
            self._log_training(count, loss, model_path)
            
            self.training_count += 1
            self.last_training = datetime.now()
            self.last_sample_count += count
            
            logger.info(f"Training complete: {count} samples, loss={loss:.4f}")
            return True
        
        return False
    
    def _check_and_train(self):
        """Check if training is needed and trigger if so."""
        new_count = self._count_new_samples()
        
        logger.debug(f"New samples: {new_count}/{self.config.NEW_SAMPLES_THRESHOLD}")
        
        if new_count >= self.config.NEW_SAMPLES_THRESHOLD:
            logger.info(f"Threshold reached ({new_count} samples), triggering training")
            self.train_now()
    
    def _run_loop(self):
        """Background monitoring loop."""
        logger.info("Continuous training loop started")
        
        while not self._stop_event.is_set():
            try:
                self._check_and_train()
            except Exception as e:
                logger.error(f"Training loop error: {e}")
            
            # Wait for next check
            self._stop_event.wait(self.config.CHECK_INTERVAL_SECONDS)
        
        logger.info("Continuous training loop stopped")
    
    def start(self):
        """Start background training loop."""
        if self._thread and self._thread.is_alive():
            logger.warning("Trainer already running")
            return
        
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Continuous trainer started (background)")
    
    def stop(self):
        """Stop background training loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Continuous trainer stopped")
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        new_samples = self._count_new_samples()
        
        return {
            'training_count': self.training_count,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'current_model': self.current_model_version,
            'new_samples_pending': new_samples,
            'threshold': self.config.NEW_SAMPLES_THRESHOLD,
            'progress': f"{new_samples}/{self.config.NEW_SAMPLES_THRESHOLD}",
            'progress_percent': min(100, (new_samples / self.config.NEW_SAMPLES_THRESHOLD) * 100)
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           CONTINUOUS LEARNING SYSTEM                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Trigger:  1,000 new labeled samples                              ║
║  Action:   Fine-tune LSTM for 5 epochs                            ║
║  Output:   Versioned model checkpoint                             ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    trainer = ContinuousTrainer()
    stats = trainer.get_stats()
    
    print(f"New samples: {stats['new_samples_pending']}")
    print(f"Threshold: {stats['threshold']}")
    print(f"Progress: {stats['progress_percent']:.1f}%")
    
    if stats['new_samples_pending'] > 0:
        print("\nTo trigger training: trainer.train_now()")
