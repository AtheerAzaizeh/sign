# OpenTrend 🔮

> Open-source Fashion Trend Forecasting System - A WGSN Alternative

## Overview

OpenTrend is an AI-powered platform that:

- **Sees** fashion through computer vision (YOLOv8 + ResNet50)
- **Listens** to market signals (Google Trends via Pytrends)
- **Thinks** by clustering similar styles (K-Means)
- **Predicts** trend trajectories (Facebook Prophet)

## Installation

```bash
# Clone and install
cd opentrend
pip install -e .

# Or install with optional features
pip install -e ".[api,scraping]"
```

## Quick Start

### Phase 1: Visual Recognition

```python
from opentrend.vision import GarmentDetector, FashionFeatureExtractor

# Detect garments
detector = GarmentDetector("yolov8n.pt")
detections = detector.detect("runway_image.jpg")

# Extract features
extractor = FashionFeatureExtractor()
features = extractor.extract("jacket.jpg")  # (2048,) vector
```

### Phase 2: Trend Signals

```python
from opentrend.signals import TrendTracker

tracker = TrendTracker()
result = tracker.analyze_keyword("cargo pants", timeframe='today 3-m')

print(result['summary'])  # 📈 'cargo pants' is RISING with +25.3% growth
print(result['metrics']['slope'])  # Weekly growth rate
```

### Phase 3: Clustering

```python
from opentrend.clustering import TrendClusterer

clusterer = TrendClusterer(n_clusters=50, use_pca=True)
labels = clusterer.fit_predict(feature_vectors)

distribution = clusterer.get_cluster_distribution()
```

### Phase 4: Forecasting

```python
from opentrend.forecasting import TrendForecaster

forecaster = TrendForecaster()
result = forecaster.fit_forecast(historical_data, periods=26)

print(result['analysis']['recommendation'])
# 🚀 STRONG BUY - High growth potential with acceptable risk
```

## Project Structure

```
opentrend/
├── vision/          # YOLOv8 + ResNet50
├── signals/         # Pytrends integration
├── clustering/      # K-Means clustering
├── forecasting/     # Prophet predictions
├── api/             # FastAPI endpoints
└── data/            # Models, vectors, signals
```

## Tech Stack

| Component   | Technology |
| ----------- | ---------- |
| Detection   | YOLOv8     |
| Features    | ResNet50   |
| Signals     | Pytrends   |
| Clustering  | K-Means    |
| Forecasting | Prophet    |

## License

MIT License - 100% Open Source
