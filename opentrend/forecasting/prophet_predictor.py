"""
opentrend/forecasting/prophet_predictor.py
Time-Series Forecasting for Fashion Trends using Facebook Prophet
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class TrendForecaster:
    """
    Predicts future trend popularity using Facebook Prophet.
    
    Why Prophet?
    - Designed for business time series with strong seasonality
    - Handles missing data gracefully (common in social data)
    - Provides intuitive uncertainty intervals
    - Easy to add custom events (Fashion Weeks, holidays)
    - Decomposable into trend + seasonality components
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        interval_width: float = 0.95
    ):
        """
        Initialize the forecaster.
        
        Args:
            yearly_seasonality: Model yearly patterns (crucial for fashion)
            weekly_seasonality: Model day-of-week effects
            daily_seasonality: Usually not needed for fashion trends
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend changes (higher = more flexible)
            interval_width: Confidence interval width (0.95 = 95%)
        """
        self.model_params = {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'interval_width': interval_width
        }
        
        self.model = None
        self.is_fitted = False
    
    def _init_model(self) -> Prophet:
        """Create a new Prophet model with configured parameters."""
        return Prophet(**self.model_params)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires specific column names).
        
        Prophet requires:
        - 'ds': datetime column
        - 'y': target value column
        
        Args:
            df: DataFrame with date and value columns
            
        Returns:
            DataFrame formatted for Prophet
        """
        df = df.copy()
        
        # Standardize date column
        if 'date' in df.columns:
            df['ds'] = pd.to_datetime(df['date'])
        elif 'ds' not in df.columns:
            raise ValueError("DataFrame must have 'date' or 'ds' column")
        
        # Standardize value column
        if 'count' in df.columns:
            df['y'] = df['count']
        elif 'frequency' in df.columns:
            df['y'] = df['frequency']
        elif 'value' in df.columns:
            df['y'] = df['value']
        elif 'interest' in df.columns:
            df['y'] = df['interest']
        elif 'y' not in df.columns:
            raise ValueError("DataFrame must have 'count', 'frequency', 'value', 'interest', or 'y' column")
        
        return df[['ds', 'y']]
    
    def fit(self, df: pd.DataFrame) -> 'TrendForecaster':
        """
        Fit the Prophet model on historical data.
        
        Args:
            df: DataFrame with date and value columns
            
        Returns:
            Self (for method chaining)
        """
        prophet_df = self.prepare_data(df)
        
        self.model = self._init_model()
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        return self
    
    def forecast(
        self,
        periods: int = 26,  # ~6 months of weekly data
        freq: str = 'W'     # Weekly frequency
    ) -> pd.DataFrame:
        """
        Generate future predictions.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()
        result.columns = ['date', 'forecast', 'lower_bound', 'upper_bound', 'trend']
        
        return result
    
    def fit_forecast(
        self,
        df: pd.DataFrame,
        periods: int = 26,
        freq: str = 'W'
    ) -> Dict:
        """
        Fit model and generate forecast in one step.
        
        Args:
            df: Historical data
            periods: Forecast horizon
            freq: Frequency
            
        Returns:
            Dict with forecast data and analysis
        """
        self.fit(df)
        forecast = self.forecast(periods, freq)
        
        prophet_df = self.prepare_data(df)
        last_date = prophet_df['ds'].max()
        
        future_forecast = forecast[forecast['date'] > last_date].copy()
        analysis = self._analyze_forecast(future_forecast, prophet_df)
        
        return {
            'full_forecast': forecast,
            'future_only': future_forecast,
            'analysis': analysis,
            'model': self.model
        }
    
    def _analyze_forecast(
        self,
        future_df: pd.DataFrame,
        historical_df: pd.DataFrame
    ) -> Dict:
        """
        Provide actionable insights from the forecast.
        
        Returns:
            Dict with trend analysis and recommendations
        """
        current_value = historical_df['y'].iloc[-1]
        peak_value = future_df['forecast'].max()
        end_value = future_df['forecast'].iloc[-1]
        peak_date = future_df.loc[future_df['forecast'].idxmax(), 'date']
        
        growth_pct = ((end_value - current_value) / max(current_value, 1)) * 100
        peak_growth_pct = ((peak_value - current_value) / max(current_value, 1)) * 100
        
        avg_uncertainty = (future_df['upper_bound'] - future_df['lower_bound']).mean()
        uncertainty_ratio = avg_uncertainty / max(future_df['forecast'].mean(), 1)
        
        if uncertainty_ratio < 0.3:
            confidence = 'high'
        elif uncertainty_ratio < 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        if growth_pct > 20 and confidence in ['high', 'medium']:
            recommendation = '🚀 STRONG BUY - High growth potential with acceptable risk'
        elif growth_pct > 10:
            recommendation = '📈 WATCH - Moderate growth expected'
        elif growth_pct > -10:
            recommendation = '➡️ HOLD - Stable with no significant change'
        else:
            recommendation = '📉 DECLINING - Consider pivoting away'
        
        return {
            'current_value': round(current_value, 2),
            'forecast_end_value': round(end_value, 2),
            'growth_percent': round(growth_pct, 1),
            'peak_value': round(peak_value, 2),
            'peak_date': peak_date.strftime('%Y-%m-%d'),
            'peak_growth_percent': round(peak_growth_pct, 1),
            'confidence': confidence,
            'uncertainty_ratio': round(uncertainty_ratio, 2),
            'recommendation': recommendation
        }
    
    def plot_forecast(
        self,
        forecast_result: Dict,
        title: str = "Fashion Trend Forecast",
        save_path: str = None
    ):
        """
        Create visualization of forecast.
        
        Args:
            forecast_result: Output from fit_forecast()
            title: Plot title
            save_path: Optional path to save figure
        """
        fig = self.model.plot(forecast_result['model'].predict(
            self.model.make_future_dataframe(periods=26, freq='W')
        ))
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Trend Interest')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        fig2 = self.model.plot_components(forecast_result['model'].predict(
            self.model.make_future_dataframe(periods=26, freq='W')
        ))
        plt.show()


# Usage Example
if __name__ == "__main__":
    # Create sample historical data
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=52, freq='W')
    trend = np.linspace(30, 70, 52)
    seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, 52))
    noise = np.random.normal(0, 5, 52)
    
    values = trend + seasonality + noise
    values = np.maximum(values, 0)
    
    historical_data = pd.DataFrame({
        'date': dates,
        'count': values
    })
    
    print("=" * 60)
    print("TREND FORECASTING DEMO")
    print("=" * 60)
    print(f"\nHistorical data: {len(historical_data)} weeks")
    print(historical_data.tail())
    
    # Initialize and forecast
    forecaster = TrendForecaster(
        yearly_seasonality=True,
        weekly_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    result = forecaster.fit_forecast(
        historical_data,
        periods=26,
        freq='W'
    )
    
    # Print analysis
    print("\n" + "=" * 60)
    print("FORECAST ANALYSIS")
    print("=" * 60)
    
    analysis = result['analysis']
    print(f"\nCurrent value: {analysis['current_value']}")
    print(f"Forecast (6 months): {analysis['forecast_end_value']}")
    print(f"Growth: {analysis['growth_percent']:+.1f}%")
    print(f"Peak: {analysis['peak_value']} on {analysis['peak_date']}")
    print(f"Confidence: {analysis['confidence'].upper()}")
    print(f"\n{analysis['recommendation']}")
    
    # Show future predictions
    print("\n" + "=" * 60)
    print("FUTURE PREDICTIONS (Next 6 Months)")
    print("=" * 60)
    print(result['future_only'][['date', 'forecast', 'lower_bound', 'upper_bound']].to_string(index=False))
