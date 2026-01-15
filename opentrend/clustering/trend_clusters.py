"""
opentrend/clustering/trend_clusters.py
K-Means Clustering for Fashion Trend Detection
"""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path


class TrendClusterer:
    """
    Groups fashion images by visual similarity using K-Means.
    
    This enables:
    - Automatic discovery of style "tribes" (e.g., all metallic jackets)
    - Tracking cluster growth over time to detect rising trends
    - Identifying outliers as potential emerging micro-trends
    
    Why K-Means?
    - Simple and interpretable
    - Scales to millions of images
    - Centroids provide "prototype" styles for each cluster
    """
    
    def __init__(
        self,
        n_clusters: int = 50,
        use_pca: bool = True,
        pca_components: int = 256,
        use_minibatch: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the clusterer.
        
        Args:
            n_clusters: Number of trend clusters (tune based on dataset)
            use_pca: Whether to reduce dimensionality before clustering
            pca_components: Target dimensions after PCA
            use_minibatch: Use MiniBatchKMeans for large datasets
            random_state: For reproducibility
        """
        self.n_clusters = n_clusters
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_minibatch = use_minibatch
        self.random_state = random_state
        
        # Will be fitted during training
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
        KMeansClass = MiniBatchKMeans if use_minibatch else KMeans
        self.kmeans = KMeansClass(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        
        self.is_fitted = False
    
    def fit(self, feature_vectors: np.ndarray, metadata: List[Dict] = None) -> 'TrendClusterer':
        """
        Fit the clustering model on feature vectors.
        
        Args:
            feature_vectors: Shape (N, 2048) from ResNet50
            metadata: Optional list of dicts with image info
            
        Returns:
            Self (for method chaining)
        """
        print(f"Fitting clusterer on {len(feature_vectors)} samples...")
        
        # Step 1: Standardize features
        X = self.scaler.fit_transform(feature_vectors)
        
        # Step 2: Reduce dimensions (optional but recommended)
        if self.use_pca:
            X = self.pca.fit_transform(X)
            print(f"  PCA: {feature_vectors.shape[1]} → {X.shape[1]} dimensions")
            print(f"  Variance retained: {sum(self.pca.explained_variance_ratio_):.2%}")
        
        # Step 3: Run K-Means
        self.kmeans.fit(X)
        self.is_fitted = True
        
        # Calculate quality metrics
        labels = self.kmeans.labels_
        silhouette = silhouette_score(X, labels, sample_size=min(10000, len(X)))
        print(f"  Silhouette score: {silhouette:.3f} (closer to 1 = better)")
        
        return self
    
    def predict(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Assign cluster IDs to new feature vectors.
        
        Args:
            feature_vectors: Shape (N, 2048)
            
        Returns:
            Array of cluster IDs, shape (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")
        
        X = self.scaler.transform(feature_vectors)
        
        if self.use_pca:
            X = self.pca.transform(X)
        
        return self.kmeans.predict(X)
    
    def fit_predict(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Fit and return cluster assignments in one step."""
        self.fit(feature_vectors)
        return self.kmeans.labels_
    
    def get_cluster_distribution(self, labels: np.ndarray = None) -> Dict[int, int]:
        """
        Get count of items in each cluster.
        
        Args:
            labels: Cluster labels. Uses training labels if None.
            
        Returns:
            Dict mapping cluster_id → count
        """
        if labels is None:
            labels = self.kmeans.labels_
        
        return dict(Counter(labels))
    
    def find_optimal_k(
        self,
        feature_vectors: np.ndarray,
        k_range: Tuple[int, int] = (10, 100),
        step: int = 10
    ) -> Dict:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            feature_vectors: Training data
            k_range: (min_k, max_k) range to search
            step: Step size for K values
            
        Returns:
            Dict with K values and their silhouette scores
        """
        # Preprocess once
        X = self.scaler.fit_transform(feature_vectors)
        if self.use_pca:
            X = self.pca.fit_transform(X)
        
        results = {}
        best_k = k_range[0]
        best_score = -1
        
        for k in range(k_range[0], k_range[1] + 1, step):
            km = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, n_init=3)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels, sample_size=min(5000, len(X)))
            results[k] = score
            
            print(f"K={k}: silhouette={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"\nOptimal K: {best_k} (silhouette: {best_score:.3f})")
        return {'scores': results, 'optimal_k': best_k}
    
    def save(self, path: str):
        """Save fitted model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'pca': self.pca,
                'kmeans': self.kmeans,
                'config': {
                    'n_clusters': self.n_clusters,
                    'use_pca': self.use_pca,
                    'pca_components': self.pca_components
                }
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'TrendClusterer':
        """Load fitted model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(**data['config'])
        instance.scaler = data['scaler']
        instance.pca = data['pca']
        instance.kmeans = data['kmeans']
        instance.is_fitted = True
        
        return instance


# Usage Example
if __name__ == "__main__":
    # Simulate feature vectors (in production, these come from ResNet50)
    np.random.seed(42)
    
    # Create synthetic clusters for demonstration
    n_samples = 5000
    true_clusters = 5
    
    # Generate clustered data
    feature_vectors = np.vstack([
        np.random.randn(n_samples // true_clusters, 2048) + i * 0.5
        for i in range(true_clusters)
    ])
    
    print("=" * 60)
    print("TREND CLUSTERING DEMO")
    print("=" * 60)
    
    # Initialize and fit clusterer
    clusterer = TrendClusterer(
        n_clusters=20,
        use_pca=True,
        pca_components=128
    )
    
    # Fit and get labels
    labels = clusterer.fit_predict(feature_vectors)
    
    # Analyze cluster distribution
    distribution = clusterer.get_cluster_distribution()
    
    print("\nCluster Distribution:")
    for cluster_id, count in sorted(distribution.items(), key=lambda x: -x[1])[:10]:
        pct = count / len(labels) * 100
        print(f"  Cluster {cluster_id}: {count} items ({pct:.1f}%)")
    
    # Save model
    clusterer.save("trend_clusterer_model.pkl")
    print("\nModel saved to 'trend_clusterer_model.pkl'")
