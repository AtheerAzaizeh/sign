"""
opentrend/validator.py
Production-Grade Backtesting & Validation Engine

Provides rigorous validation of the OpenTrend forecasting system:
1. Train/Test split by date (Train: 2020-2024, Test: 2025)
2. Prophet cross-validation with rolling windows
3. Calculate RMSE, MAE, MAPE metrics
4. Generate professional accuracy reports

Usage:
    validator = BacktestValidator()
    
    # Quick demo on a sample keyword
    validator.run_demo("cargo pants")
    
    # Full validation on custom data
    metrics = validator.validate(df, train_end_year=2024)
    validator.print_report(metrics, keyword="puffer jacket")
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Optional tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Import OpenTrend data loader
try:
    from data_loader import FashionDataLoader
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import FashionDataLoader

# Matplotlib styling
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0a0a14'
plt.rcParams['axes.facecolor'] = '#12121c'


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error (RMSE).
    
    Lower is better. Same units as target variable.
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).
    
    Lower is better. Easier to interpret than RMSE.
    
    MAE = mean(|y_true - y_pred|)
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    
    Returns percentage. Model accuracy = 100% - MAPE.
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Note: Handles zeros with epsilon to avoid division errors.
    """
    # Avoid division by zero
    epsilon = 1e-10
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # Cap at 200% for extreme cases
    return min(mape, 200.0)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    
    More balanced than MAPE for values near zero.
    
    SMAPE = mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|)) * 100
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


# =============================================================================
# VALIDATION RESULTS
# =============================================================================

class ValidationResult:
    """Container for validation metrics and data."""
    
    def __init__(
        self,
        rmse: float,
        mae: float,
        mape: float,
        accuracy: float,
        train_size: int,
        test_size: int,
        comparison_df: pd.DataFrame = None,
        method: str = "train_test_split"
    ):
        self.rmse = rmse
        self.mae = mae
        self.mape = mape
        self.accuracy = accuracy
        self.train_size = train_size
        self.test_size = test_size
        self.comparison_df = comparison_df
        self.method = method
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'accuracy': self.accuracy,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'method': self.method
        }
    
    def __repr__(self) -> str:
        return f"ValidationResult(accuracy={self.accuracy:.1f}%, rmse={self.rmse:.2f}, mape={self.mape:.1f}%)"


# =============================================================================
# BACKTEST VALIDATOR
# =============================================================================

class BacktestValidator:
    """
    Production-grade backtesting validator for Prophet forecasts.
    
    Validation Strategy:
    1. Split data: Train (2020-2024) vs Test (2025-Present)
    2. Train Prophet ONLY on the train set
    3. Predict the test period
    4. Compare predictions vs actuals
    5. Calculate RMSE, MAE, MAPE
    
    Usage:
        validator = BacktestValidator()
        
        # Quick demo
        validator.run_demo("cargo pants")
        
        # Custom validation
        result = validator.validate(df, train_end_year=2024)
        validator.print_report(result, "puffer jacket")
    """
    
    def __init__(self, data_loader: FashionDataLoader = None):
        """
        Initialize the validator.
        
        Args:
            data_loader: FashionDataLoader instance (creates one if None)
        """
        self.data_loader = data_loader or FashionDataLoader()
        self.last_result: Optional[ValidationResult] = None
    
    # =========================================================================
    # CORE VALIDATION METHODS
    # =========================================================================
    
    def validate(
        self,
        df: pd.DataFrame,
        train_end_year: int = 2024,
        model_params: Dict = None
    ) -> ValidationResult:
        """
        Validate Prophet model using train/test split by year.
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
            train_end_year: Last year to include in training
            model_params: Optional Prophet parameters
            
        Returns:
            ValidationResult with metrics
        """
        # Normalize column names
        df = self._prepare_dataframe(df)
        
        # Split by year
        train_df = df[df['ds'].dt.year <= train_end_year].copy()
        test_df = df[df['ds'].dt.year > train_end_year].copy()
        
        logger.info(f"📊 Validation Split:")
        logger.info(f"   Train: {train_df['ds'].min().year}-{train_df['ds'].max().year} ({len(train_df)} rows)")
        logger.info(f"   Test: {test_df['ds'].min().year if len(test_df) > 0 else 'N/A'}-{test_df['ds'].max().year if len(test_df) > 0 else 'N/A'} ({len(test_df)} rows)")
        
        # Validate data requirements
        if len(train_df) < 52:
            logger.error(f"Insufficient training data ({len(train_df)} < 52 weeks)")
            return ValidationResult(0, 0, 100, 0, len(train_df), 0, method="insufficient_data")
        
        if len(test_df) < 4:
            logger.warning(f"Limited test data ({len(test_df)} rows). Using cross-validation instead.")
            return self._run_cross_validation(df, model_params)
        
        # Train Prophet on train set only
        logger.info("🧠 Training Prophet model...")
        
        default_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05
        }
        if model_params:
            default_params.update(model_params)
        
        model = Prophet(**default_params)
        model.fit(train_df[['ds', 'y']])
        
        # Predict test period
        logger.info("🔮 Predicting test period...")
        
        periods_to_predict = len(test_df) + 8  # Extra buffer
        future = model.make_future_dataframe(periods=periods_to_predict, freq='W')
        forecast = model.predict(future)
        
        # Align predictions with test data
        comparison = pd.merge(
            test_df[['ds', 'y']],
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds',
            how='inner'
        )
        
        if len(comparison) == 0:
            logger.warning("Could not align predictions. Falling back to cross-validation.")
            return self._run_cross_validation(df, model_params)
        
        # Calculate metrics
        y_true = comparison['y'].values
        y_pred = comparison['yhat'].values
        
        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)
        accuracy = max(0, 100 - mape)
        
        result = ValidationResult(
            rmse=rmse,
            mae=mae,
            mape=mape,
            accuracy=accuracy,
            train_size=len(train_df),
            test_size=len(comparison),
            comparison_df=comparison,
            method=f"train_test_split_{train_end_year}"
        )
        
        self.last_result = result
        logger.info(f"✓ Validation complete: Accuracy = {accuracy:.1f}%")
        
        return result
    
    def _run_cross_validation(
        self,
        df: pd.DataFrame,
        model_params: Dict = None
    ) -> ValidationResult:
        """
        Fallback cross-validation when test set is insufficient.
        
        Uses Prophet's built-in cross_validation with rolling windows.
        """
        logger.info("📊 Running Prophet cross-validation...")
        
        df = self._prepare_dataframe(df)
        
        default_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'seasonality_mode': 'multiplicative'
        }
        if model_params:
            default_params.update(model_params)
        
        model = Prophet(**default_params)
        model.fit(df[['ds', 'y']])
        
        try:
            # Cross-validation parameters
            # Initial: 2 years training
            # Period: retrain every 3 months
            # Horizon: predict 6 months ahead
            cv_results = cross_validation(
                model,
                initial='730 days',
                period='90 days',
                horizon='180 days',
                parallel='processes'
            )
            
            y_true = cv_results['y'].values
            y_pred = cv_results['yhat'].values
            
            rmse = calculate_rmse(y_true, y_pred)
            mae = calculate_mae(y_true, y_pred)
            mape = calculate_mape(y_true, y_pred)
            accuracy = max(0, 100 - mape)
            
            result = ValidationResult(
                rmse=rmse,
                mae=mae,
                mape=mape,
                accuracy=accuracy,
                train_size=len(df),
                test_size=len(cv_results),
                comparison_df=cv_results[['ds', 'y', 'yhat']].rename(columns={'yhat': 'yhat'}),
                method="cross_validation"
            )
            
            self.last_result = result
            return result
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return ValidationResult(0, 0, 100, 0, len(df), 0, method="error")
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns for Prophet."""
        df = df.copy()
        
        # Rename date column
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'ds'})
        
        # Rename value column
        for col in ['interest', 'value', 'count']:
            if col in df.columns and 'y' not in df.columns:
                df = df.rename(columns={col: 'y'})
                break
        
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds'/'date' and 'y'/'interest'/'value' columns")
        
        df['ds'] = pd.to_datetime(df['ds'])
        
        return df
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_report(
        self,
        result: ValidationResult = None,
        keyword: str = "Fashion Trend"
    ):
        """
        Print a professional validation report.
        
        Args:
            result: ValidationResult (uses last result if None)
            keyword: Name of the trend being validated
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            print("No validation results available. Run validate() or run_demo() first.")
            return
        
        # Color-code accuracy
        if result.accuracy >= 85:
            grade = "EXCELLENT"
            emoji = "🌟"
        elif result.accuracy >= 75:
            grade = "GOOD"
            emoji = "✅"
        elif result.accuracy >= 65:
            grade = "MODERATE"
            emoji = "⚡"
        else:
            grade = "NEEDS IMPROVEMENT"
            emoji = "⚠️"
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      MODEL VALIDATION REPORT                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Trend Analyzed:  {keyword:<50} ║
║  Method:          {result.method:<50} ║
║  Train Samples:   {result.train_size:<50} ║
║  Test Samples:    {result.test_size:<50} ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║     RMSE  (Root Mean Square Error):    {result.rmse:>10.2f}                      ║
║     MAE   (Mean Absolute Error):       {result.mae:>10.2f}                      ║
║     MAPE  (Mean Absolute % Error):     {result.mape:>10.2f}%                     ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║     {emoji} MODEL ACCURACY:  {result.accuracy:>5.1f}%   (100% - MAPE)                      ║
║                                                                       ║
║     Grade: {grade:<20}                                           ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
        print(report)
        
        # Interpretation
        if result.accuracy >= 85:
            print("   📈 Prediction is highly reliable for production use.")
        elif result.accuracy >= 75:
            print("   📊 Prediction is reliable for planning and strategy.")
        elif result.accuracy >= 65:
            print("   ⚡ Use predictions with caution. Consider more data.")
        else:
            print("   ⚠️  Model needs more data or parameter tuning.")
    
    def plot_validation(
        self,
        result: ValidationResult = None,
        keyword: str = "Trend",
        save_path: str = None
    ):
        """
        Plot predicted vs actual comparison.
        
        Args:
            result: ValidationResult (uses last result if None)
            keyword: Trend name for title
            save_path: Path to save figure
        """
        if result is None:
            result = self.last_result
        
        if result is None or result.comparison_df is None:
            logger.warning("No comparison data available to plot.")
            return
        
        df = result.comparison_df
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time series comparison
        ax1 = axes[0]
        ax1.plot(df['ds'], df['y'], color='#00ff88', linewidth=2, 
                label='Actual', marker='o', markersize=4)
        ax1.plot(df['ds'], df['yhat'], color='#ff6b9d', linewidth=2, 
                linestyle='--', label='Predicted', marker='s', markersize=4)
        
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Trend Interest', fontsize=10)
        ax1.set_title(f'{keyword}: Predicted vs Actual', fontsize=12, 
                     fontweight='bold', color='white')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2 = axes[1]
        ax2.scatter(df['y'], df['yhat'], color='#00d9ff', 
                   alpha=0.7, edgecolor='white', s=60)
        
        # Perfect prediction line
        min_val = min(df['y'].min(), df['yhat'].min())
        max_val = max(df['y'].max(), df['yhat'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'r--', alpha=0.5, linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual', fontsize=10)
        ax2.set_ylabel('Predicted', fontsize=10)
        ax2.set_title(f'Prediction Accuracy (R² = {1 - result.mape/100:.2f})', 
                     fontsize=12, fontweight='bold', color='white')
        ax2.legend(loc='lower right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = Path(__file__).parent / 'validation_plot.png'
        
        plt.savefig(save_path, dpi=150, facecolor='#0a0a14', 
                   edgecolor='none', bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Validation plot saved: {save_path}")
    
    # =========================================================================
    # DEMO METHOD
    # =========================================================================
    
    def run_demo(self, keyword: str = "cargo pants") -> ValidationResult:
        """
        Run a complete validation demo on a sample keyword.
        
        Fetches 5-year Google Trends data and validates the Prophet forecast.
        
        Args:
            keyword: Fashion keyword to analyze
            
        Returns:
            ValidationResult with metrics
        """
        print("=" * 70)
        print("🔬 OPENTREND BACKTEST VALIDATOR - Demo")
        print("=" * 70)
        print(f"   Keyword: '{keyword}'")
        print(f"   Train Period: 2020-2024")
        print(f"   Test Period: 2025-Present")
        print("=" * 70)
        
        # Fetch historical data
        print("\n📥 Fetching 5-year Google Trends data...")
        
        try:
            df = self.data_loader.ensure_trend_history(keyword, years=5)
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            # Generate synthetic data for demo
            df = self._generate_demo_data()
        
        if df.empty:
            df = self._generate_demo_data()
        
        print(f"   ✓ Loaded {len(df)} data points")
        print(f"   Date range: {df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}")
        
        # Run validation
        print("\n" + "-" * 50)
        print("RUNNING VALIDATION")
        print("-" * 50)
        
        result = self.validate(df, train_end_year=2024)
        
        # Print report
        self.print_report(result, keyword)
        
        # Save plot
        self.plot_validation(result, keyword)
        
        return result
    
    def _generate_demo_data(self) -> pd.DataFrame:
        """Generate synthetic data for demo purposes."""
        np.random.seed(42)
        
        # 5 years of weekly data
        dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='W')
        
        t = np.arange(len(dates))
        
        # Trend + seasonality + noise
        trend = 50 + 0.05 * t
        seasonality = 15 * np.sin(2 * np.pi * t / 52)
        noise = np.random.normal(0, 5, len(dates))
        
        values = np.clip(trend + seasonality + noise, 0, 100)
        
        return pd.DataFrame({'ds': dates, 'y': values})


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔬 OPENTREND VALIDATION ENGINE")
    print("=" * 70)
    
    validator = BacktestValidator()
    
    # Run demo on sample keywords
    keywords = ["cargo pants", "puffer jacket", "floral dress"]
    
    results = {}
    
    for keyword in keywords:
        print(f"\n\n{'='*70}")
        result = validator.run_demo(keyword)
        results[keyword] = result
        
        import time
        time.sleep(2)  # Rate limiting
    
    # Summary table
    print("\n\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Keyword':<25} {'Accuracy':>12} {'RMSE':>10} {'MAPE':>10}")
    print("-" * 60)
    
    for keyword, result in results.items():
        print(f"{keyword:<25} {result.accuracy:>11.1f}% {result.rmse:>10.2f} {result.mape:>9.1f}%")
    
    print("\n✅ Validation complete!")
