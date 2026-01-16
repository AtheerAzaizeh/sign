"""
opentrend/real_forecaster.py
Production-Grade Integrated Forecasting Pipeline

The "Main Brain" connecting:
    Vision (ResNet50) → Clustering (K-Means) → Signals (Google Trends) → Forecasting (Prophet)

Pipeline Flow:
1. Load fashion images from dataset
2. Extract 2048-dim feature vectors using ResNet50
3. Cluster features to discover "Visual Micro-Trends"
4. Label each cluster by its dominant ground-truth label
5. Fetch 5-year Google Trends history for each label
6. Forecast next 6 months using Facebook Prophet

Output: List of MicroTrend objects with growth predictions and forecast charts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional tqdm import
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Import existing OpenTrend modules
try:
    from vision.feature_extractor import FashionFeatureExtractor
    from clustering.trend_clusters import TrendClusterer
    from forecasting.prophet_predictor import TrendForecaster
    from data_loader import FashionDataLoader, FashionDataset, FashionItem
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vision.feature_extractor import FashionFeatureExtractor
    from clustering.trend_clusters import TrendClusterer
    from forecasting.prophet_predictor import TrendForecaster
    from data_loader import FashionDataLoader, FashionDataset, FashionItem

# Set plotting style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0a0a14'
plt.rcParams['axes.facecolor'] = '#12121c'
plt.rcParams['axes.edgecolor'] = '#333333'


# =============================================================================
# MICRO-TREND DATA STRUCTURE
# =============================================================================

@dataclass
class MicroTrend:
    """
    Represents a discovered micro-trend (visual cluster) with its forecast.
    
    A micro-trend is a group of visually similar fashion items that share
    common characteristics, labeled by the dominant category in the cluster.
    
    Attributes:
        cluster_id: Unique cluster identifier
        dominant_label: Most common ground-truth label in cluster
        label_confidence: Fraction of items matching dominant label
        item_count: Number of items in cluster
        representative_images: Paths to images closest to centroid
        trend_history: 5-year Google Trends DataFrame
        forecast: 6-month Prophet forecast DataFrame
        analysis: Growth metrics and recommendations
    """
    cluster_id: int
    dominant_label: str
    label_confidence: float
    item_count: int
    
    representative_images: List[str] = field(default_factory=list)
    trend_history: Optional[pd.DataFrame] = None
    forecast: Optional[pd.DataFrame] = None
    analysis: Optional[Dict[str, Any]] = None
    
    @property
    def growth_rate(self) -> float:
        """Get predicted growth rate percentage."""
        if self.analysis:
            return self.analysis.get('growth_percent', 0.0)
        return 0.0
    
    @property
    def recommendation(self) -> str:
        """Get actionable recommendation."""
        if self.analysis:
            return self.analysis.get('recommendation', 'No analysis available')
        return 'Forecast not available'
    
    @property
    def confidence_level(self) -> str:
        """Get forecast confidence level."""
        if self.analysis:
            return self.analysis.get('confidence', 'unknown')
        return 'unknown'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'dominant_label': self.dominant_label,
            'label_confidence': self.label_confidence,
            'item_count': self.item_count,
            'growth_rate': self.growth_rate,
            'recommendation': self.recommendation,
            'confidence_level': self.confidence_level
        }
    
    def __repr__(self) -> str:
        return (f"MicroTrend(id={self.cluster_id}, label='{self.dominant_label}', "
                f"items={self.item_count}, growth={self.growth_rate:+.1f}%)")


# =============================================================================
# INTEGRATED FORECAST PIPELINE
# =============================================================================

class IntegratedForecastPipeline:
    """
    Production-grade pipeline connecting Vision → Clustering → Signals → Forecasting.
    
    This is the "Main Brain" of OpenTrend that:
    1. Loads fashion images
    2. Extracts visual features (ResNet50)
    3. Clusters into micro-trends (K-Means)
    4. Labels by dominant ground-truth category
    5. Fetches Google Trends history
    6. Forecasts next 6 months (Prophet)
    
    Usage:
        pipeline = IntegratedForecastPipeline(n_clusters=20)
        
        # Run on dataset
        dataset = loader.load_dataset("./data/fashion")
        micro_trends = pipeline.run(dataset)
        
        # Or run demo
        micro_trends = pipeline.run_demo()
        
        # Generate report
        pipeline.print_report(micro_trends)
    """
    
    def __init__(
        self,
        n_clusters: int = 20,
        trend_years: int = 5,
        forecast_periods: int = 26,  # ~6 months weekly
        device: str = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the integrated pipeline.
        
        Args:
            n_clusters: Number of micro-trend clusters to discover
            trend_years: Years of historical Google Trends data
            forecast_periods: Weeks to forecast ahead (26 = 6 months)
            device: 'cuda' or 'cpu' for ResNet50 inference
            cache_enabled: Enable caching for Google Trends
        """
        self.n_clusters = n_clusters
        self.trend_years = trend_years
        self.forecast_periods = forecast_periods
        self.device = device
        
        # Lazy initialization for heavy components
        self._feature_extractor = None
        self._clusterer = None
        self._forecaster = None
        self._data_loader = None
        
        self.cache_enabled = cache_enabled
        
        logger.info(f"Pipeline initialized: {n_clusters} clusters, {trend_years}y history, {forecast_periods}w forecast")
    
    # =========================================================================
    # LAZY COMPONENT INITIALIZATION
    # =========================================================================
    
    @property
    def feature_extractor(self) -> FashionFeatureExtractor:
        """Lazy init of ResNet50 feature extractor."""
        if self._feature_extractor is None:
            logger.info("🔧 Loading ResNet50 feature extractor...")
            self._feature_extractor = FashionFeatureExtractor(device=self.device)
        return self._feature_extractor
    
    @property
    def clusterer(self) -> TrendClusterer:
        """Lazy init of K-Means clusterer."""
        if self._clusterer is None:
            self._clusterer = TrendClusterer(
                n_clusters=self.n_clusters,
                use_pca=True,
                pca_components=256,
                use_minibatch=True
            )
        return self._clusterer
    
    @property
    def forecaster(self) -> TrendForecaster:
        """Lazy init of Prophet forecaster."""
        if self._forecaster is None:
            self._forecaster = TrendForecaster(
                yearly_seasonality=True,
                weekly_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )
        return self._forecaster
    
    @property
    def data_loader(self) -> FashionDataLoader:
        """Lazy init of data loader."""
        if self._data_loader is None:
            self._data_loader = FashionDataLoader(cache_ttl_days=7)
        return self._data_loader
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run(
        self,
        dataset: FashionDataset,
        show_progress: bool = True
    ) -> List[MicroTrend]:
        """
        Run the full integrated pipeline on a dataset.
        
        Args:
            dataset: FashionDataset loaded via FashionDataLoader
            show_progress: Show progress bars
            
        Returns:
            List of MicroTrend objects sorted by growth potential
        """
        logger.info("=" * 60)
        logger.info("🚀 OPENTREND INTEGRATED FORECAST PIPELINE")
        logger.info("=" * 60)
        logger.info(f"   Dataset: {len(dataset)} items")
        logger.info(f"   Labels: {len(dataset.unique_labels)} unique categories")
        logger.info(f"   Clusters: {self.n_clusters}")
        logger.info(f"   Trend History: {self.trend_years} years")
        logger.info(f"   Forecast: {self.forecast_periods} weeks")
        logger.info("=" * 60)
        
        # Step 1: Extract visual features
        logger.info("\n📷 Step 1: Extracting visual features (ResNet50)...")
        features, valid_items = self._extract_features(dataset, show_progress)
        
        if len(features) == 0:
            logger.error("No valid features extracted. Check image paths.")
            return []
        
        # Step 2: Cluster into micro-trends
        logger.info(f"\n🎯 Step 2: Clustering into {self.n_clusters} micro-trends (K-Means)...")
        cluster_labels = self._cluster_features(features)
        
        # Step 3: Determine dominant labels per cluster
        logger.info("\n🏷️  Step 3: Determining dominant labels per cluster...")
        micro_trends = self._create_labeled_clusters(valid_items, cluster_labels, features)
        
        # Step 4: Fetch Google Trends history
        logger.info("\n📈 Step 4: Fetching Google Trends history...")
        self._fetch_trends(micro_trends, show_progress)
        
        # Step 5: Run Prophet forecasts
        logger.info("\n🔮 Step 5: Running Prophet forecasts...")
        self._run_forecasts(micro_trends, show_progress)
        
        # Sort by growth rate
        micro_trends.sort(key=lambda mt: mt.growth_rate, reverse=True)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return micro_trends
    
    def run_demo(self, n_items: int = 50) -> List[MicroTrend]:
        """
        Run demo pipeline with synthetic visual clusters but real Google Trends.
        
        Useful for testing without a real image dataset.
        
        Args:
            n_items: Simulated number of items per cluster
            
        Returns:
            List of MicroTrend objects
        """
        logger.info("=" * 60)
        logger.info("🎭 OPENTREND DEMO MODE")
        logger.info("   Using synthetic clusters with real Google Trends")
        logger.info("=" * 60)
        
        # Simulate fashion categories as discovered clusters
        demo_categories = [
            ("puffer_jacket", 85, 0.92),
            ("cargo_pants", 72, 0.88),
            ("floral_dress", 65, 0.85),
            ("oversized_hoodie", 58, 0.90),
            ("mini_skirt", 45, 0.78),
            ("leather_jacket", 55, 0.86),
            ("wide_leg_jeans", 68, 0.82),
            ("cropped_blazer", 42, 0.75),
            ("midi_dress", 50, 0.80),
            ("denim_jacket", 62, 0.88),
        ]
        
        # Limit to n_clusters
        demo_categories = demo_categories[:min(self.n_clusters, len(demo_categories))]
        
        micro_trends = []
        
        iterator = tqdm(demo_categories, desc="Processing clusters")
        for idx, (label, item_count, confidence) in enumerate(iterator):
            iterator.set_postfix_str(label)
            
            micro_trend = MicroTrend(
                cluster_id=idx,
                dominant_label=label,
                label_confidence=confidence,
                item_count=item_count
            )
            
            # Fetch real Google Trends
            try:
                df = self.data_loader.ensure_trend_history(label, years=self.trend_years)
                micro_trend.trend_history = df
            except Exception as e:
                logger.warning(f"Could not fetch trends for '{label}': {e}")
            
            micro_trends.append(micro_trend)
        
        # Run forecasts
        logger.info("\n🔮 Running Prophet forecasts...")
        self._run_forecasts(micro_trends, show_progress=True)
        
        # Sort by growth
        micro_trends.sort(key=lambda mt: mt.growth_rate, reverse=True)
        
        return micro_trends
    
    # =========================================================================
    # PIPELINE STEPS
    # =========================================================================
    
    def _extract_features(
        self,
        dataset: FashionDataset,
        show_progress: bool
    ) -> Tuple[np.ndarray, List[FashionItem]]:
        """
        Extract ResNet50 features from images.
        
        Returns:
            Tuple of (feature_vectors, valid_items)
        """
        valid_items = []
        valid_paths = []
        
        for item in dataset:
            if Path(item.image_path).exists():
                valid_items.append(item)
                valid_paths.append(item.image_path)
        
        if not valid_paths:
            logger.warning("No valid image paths found!")
            return np.array([]), []
        
        logger.info(f"   Processing {len(valid_paths)} images...")
        
        # Extract features in batches
        batch_size = 32
        all_features = []
        
        iterator = range(0, len(valid_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")
        
        for i in iterator:
            batch = valid_paths[i:i + batch_size]
            try:
                features = self.feature_extractor.extract_batch(batch)
                all_features.append(features)
            except Exception as e:
                logger.error(f"Feature extraction error: {e}")
                # Append zeros for failed batch
                all_features.append(np.zeros((len(batch), 2048)))
        
        features = np.vstack(all_features)
        logger.info(f"   ✓ Extracted {features.shape[0]} feature vectors (2048-dim)")
        
        return features, valid_items
    
    def _cluster_features(self, features: np.ndarray) -> np.ndarray:
        """
        Cluster feature vectors using K-Means.
        
        Returns:
            Array of cluster labels
        """
        labels = self.clusterer.fit_predict(features)
        
        # Log distribution
        distribution = Counter(labels)
        top_3 = distribution.most_common(3)
        logger.info(f"   ✓ Created {len(distribution)} clusters")
        logger.info(f"   Top 3 clusters: {[(f'C{c}', n) for c, n in top_3]}")
        
        return labels
    
    def _create_labeled_clusters(
        self,
        items: List[FashionItem],
        cluster_labels: np.ndarray,
        features: np.ndarray
    ) -> List[MicroTrend]:
        """
        Create MicroTrend objects with dominant labels for each cluster.
        
        For each cluster, find the most common ground-truth label
        (e.g., if 80% of items are "puffer_jacket", that's the dominant label).
        """
        micro_trends = []
        
        for cluster_id in range(self.n_clusters):
            # Get items in this cluster
            mask = cluster_labels == cluster_id
            cluster_items = [items[i] for i in range(len(items)) if mask[i]]
            
            if not cluster_items:
                continue
            
            # Find dominant label
            labels = [item.ground_truth_label for item in cluster_items]
            label_counts = Counter(labels)
            dominant_label, count = label_counts.most_common(1)[0]
            confidence = count / len(cluster_items)
            
            # Find representative images (closest to centroid)
            cluster_features = features[mask]
            centroid = cluster_features.mean(axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_idx = np.argsort(distances)[:5]
            representative_images = [cluster_items[i].image_path for i in closest_idx]
            
            micro_trend = MicroTrend(
                cluster_id=cluster_id,
                dominant_label=dominant_label,
                label_confidence=confidence,
                item_count=len(cluster_items),
                representative_images=representative_images
            )
            
            micro_trends.append(micro_trend)
            logger.info(f"   Cluster {cluster_id}: '{dominant_label}' ({confidence:.0%} confidence, {len(cluster_items)} items)")
        
        return micro_trends
    
    def _fetch_trends(
        self,
        micro_trends: List[MicroTrend],
        show_progress: bool
    ):
        """Fetch Google Trends history for each micro-trend."""
        iterator = micro_trends
        if show_progress:
            iterator = tqdm(micro_trends, desc="Fetching trends")
        
        for mt in iterator:
            if show_progress:
                iterator.set_postfix_str(mt.dominant_label[:15])
            
            try:
                df = self.data_loader.ensure_trend_history(
                    mt.dominant_label,
                    years=self.trend_years
                )
                mt.trend_history = df
            except Exception as e:
                logger.warning(f"Failed to fetch trends for '{mt.dominant_label}': {e}")
    
    def _run_forecasts(
        self,
        micro_trends: List[MicroTrend],
        show_progress: bool
    ):
        """Run Prophet forecasts for each micro-trend."""
        iterator = micro_trends
        if show_progress:
            iterator = tqdm(micro_trends, desc="Forecasting")
        
        for mt in iterator:
            if mt.trend_history is None or mt.trend_history.empty:
                continue
            
            if show_progress:
                iterator.set_postfix_str(mt.dominant_label[:15])
            
            try:
                # Prepare data for Prophet
                df = mt.trend_history[['ds', 'y']].copy()
                
                # Run forecast
                result = self.forecaster.fit_forecast(
                    df,
                    periods=self.forecast_periods,
                    freq='W'
                )
                
                mt.forecast = result['future_only']
                mt.analysis = result['analysis']
                
            except Exception as e:
                logger.warning(f"Forecast failed for '{mt.dominant_label}': {e}")
    
    # =========================================================================
    # REPORTING & VISUALIZATION
    # =========================================================================
    
    def print_report(self, micro_trends: List[MicroTrend]):
        """Print a professional forecast report."""
        print("\n" + "=" * 70)
        print("          OPENTREND MICRO-TREND FORECAST REPORT")
        print(f"                {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)
        
        # Rising trends
        rising = [mt for mt in micro_trends if mt.growth_rate > 0][:10]
        
        print("\n🚀 TOP RISING MICRO-TRENDS (Next 6 Months)")
        print("-" * 60)
        print(f"{'Rank':<5} {'Label':<25} {'Growth':>10} {'Items':>8} {'Conf':>8}")
        print("-" * 60)
        
        for i, mt in enumerate(rising, 1):
            label = mt.dominant_label.replace('_', ' ').title()[:24]
            print(f"{i:<5} {label:<25} {mt.growth_rate:>+9.1f}% {mt.item_count:>8} {mt.label_confidence:>7.0%}")
        
        # Declining trends
        declining = [mt for mt in micro_trends if mt.growth_rate < -5][:5]
        
        if declining:
            print("\n📉 DECLINING TRENDS (Consider Avoiding)")
            print("-" * 60)
            
            for mt in declining:
                label = mt.dominant_label.replace('_', ' ').title()
                print(f"   ⬇️ {label:<30} {mt.growth_rate:>+.1f}%")
        
        # Top recommendation
        if rising:
            top = rising[0]
            print("\n" + "=" * 70)
            print("   🎯 TOP RECOMMENDATION")
            print("=" * 70)
            print(f"""
   Trend:       {top.dominant_label.replace('_', ' ').title()}
   Growth:      {top.growth_rate:+.1f}%
   Confidence:  {top.confidence_level.upper()}
   Items:       {top.item_count} in cluster
   
   {top.recommendation}
""")
        
        print("=" * 70)
    
    def create_dashboard(
        self,
        micro_trends: List[MicroTrend],
        top_n: int = 6,
        save_path: str = None
    ) -> str:
        """
        Create visual dashboard of top micro-trends.
        
        Args:
            micro_trends: List of MicroTrend objects
            top_n: Number of top trends to show
            save_path: Path to save figure
            
        Returns:
            Path to saved dashboard
        """
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('OPENTREND MICRO-TREND FORECAST DASHBOARD', 
                     fontsize=16, fontweight='bold', color='white', y=0.98)
        
        # Get top trends with forecasts
        valid_trends = [mt for mt in micro_trends 
                       if mt.forecast is not None and mt.trend_history is not None][:top_n]
        
        if not valid_trends:
            logger.warning("No trends with forecasts to visualize")
            return None
        
        rows = (len(valid_trends) + 2) // 3
        cols = min(3, len(valid_trends))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(valid_trends)))
        
        for idx, (mt, color) in enumerate(zip(valid_trends, colors)):
            ax = fig.add_subplot(rows, cols, idx + 1)
            
            # Plot history
            ax.plot(mt.trend_history['ds'], mt.trend_history['y'],
                   color='#666666', alpha=0.6, linewidth=1, label='History')
            
            # Plot forecast
            ax.plot(mt.forecast['date'], mt.forecast['forecast'],
                   color=color, linewidth=2.5, label='Forecast')
            
            # Confidence interval
            ax.fill_between(
                mt.forecast['date'],
                mt.forecast['lower_bound'],
                mt.forecast['upper_bound'],
                alpha=0.25, color=color
            )
            
            # Title with growth
            growth_color = '#00ff88' if mt.growth_rate > 10 else '#ffd93d' if mt.growth_rate > 0 else '#ff6b6b'
            title = mt.dominant_label.replace('_', ' ').title()
            ax.set_title(f"{title}\n{mt.growth_rate:+.0f}%", 
                        fontsize=10, fontweight='bold', color=growth_color)
            
            ax.grid(True, alpha=0.2)
            ax.tick_params(colors='#888888', labelsize=8)
            ax.set_ylabel('Interest', fontsize=8, color='#888888')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        if save_path is None:
            save_path = Path(__file__).parent / 'microtrend_dashboard.png'
        
        plt.savefig(save_path, dpi=150, facecolor='#0a0a14', 
                   edgecolor='none', bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Dashboard saved: {save_path}")
        return str(save_path)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 OPENTREND INTEGRATED FORECAST PIPELINE")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = IntegratedForecastPipeline(
        n_clusters=10,
        trend_years=5,
        forecast_periods=26
    )
    
    # Run demo mode
    micro_trends = pipeline.run_demo()
    
    # Print report
    pipeline.print_report(micro_trends)
    
    # Create dashboard
    pipeline.create_dashboard(micro_trends)
    
    print("\n✅ Pipeline demo complete!")
    print("""
    To run with real images:
    
        from data_loader import FashionDataLoader
        from real_forecaster import IntegratedForecastPipeline
        
        loader = FashionDataLoader()
        dataset = loader.load_dataset("./data/fashion")
        
        pipeline = IntegratedForecastPipeline(n_clusters=20)
        micro_trends = pipeline.run(dataset)
        
        pipeline.print_report(micro_trends)
        pipeline.create_dashboard(micro_trends)
    """)
