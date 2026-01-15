"""
opentrend/vision/detector.py
YOLOv8 Garment Detection Pipeline
"""
from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import List, Dict


class GarmentDetector:
    """
    Detects garment types in images using YOLOv8.
    
    For production, fine-tune on DeepFashion2 dataset.
    Pre-trained model detects: dress, jacket, pants, shirt, skirt, etc.
    """
    
    # DeepFashion2 class mappings
    GARMENT_CLASSES = {
        0: 'short_sleeve_top', 1: 'long_sleeve_top', 2: 'short_sleeve_outwear',
        3: 'long_sleeve_outwear', 4: 'vest', 5: 'sling', 6: 'shorts',
        7: 'trousers', 8: 'skirt', 9: 'short_sleeve_dress',
        10: 'long_sleeve_dress', 11: 'vest_dress', 12: 'sling_dress'
    }
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize detector with YOLOv8 model.
        
        Args:
            model_path: Path to model weights. Use custom trained
                       model for fashion-specific detection.
        """
        self.model = YOLO(model_path)
    
    def detect(self, image_path: str, confidence: float = 0.5) -> List[Dict]:
        """
        Detect garments in an image.
        
        Args:
            image_path: Path to input image
            confidence: Minimum confidence threshold
            
        Returns:
            List of detections with bounding boxes and class info
        """
        results = self.model(image_path, conf=confidence)[0]
        
        detections = []
        for box in results.boxes:
            detection = {
                'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'class_name': results.names[int(box.cls[0])]
            }
            detections.append(detection)
        
        return detections
    
    def extract_crops(self, image_path: str, detections: List[Dict]) -> List[Image.Image]:
        """
        Extract cropped regions for each detected garment.
        
        Args:
            image_path: Path to original image
            detections: List of detection dictionaries
            
        Returns:
            List of PIL Images (cropped garments)
        """
        image = Image.open(image_path).convert('RGB')
        crops = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        
        return crops


# Usage Example
if __name__ == "__main__":
    detector = GarmentDetector("path/to/fashion_yolov8.pt")
    
    detections = detector.detect("runway_image.jpg")
    crops = detector.extract_crops("runway_image.jpg", detections)
    
    for i, (det, crop) in enumerate(zip(detections, crops)):
        print(f"Detected: {det['class_name']} (conf: {det['confidence']:.2f})")
        crop.save(f"garment_{i}_{det['class_name']}.jpg")
