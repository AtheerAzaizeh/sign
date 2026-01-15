"""
opentrend/signals/trend_tracker.py
Google Trends Signal Tracking with Growth Analysis
"""
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time


class TrendTracker:
    """
    Tracks Google Trends data and calculates growth metrics.
    
    Why Pytrends?
    - Free access to Google Trends data
    - No API key required
    - Captures real consumer search behavior
    - Weekly granularity perfect for fashion cycles
    """
    
    def __init__(self, hl: str = 'en-US', tz: int = 360, retries: int = 3, backoff_factor: float = 1.0):
        """
        Initialize the Pytrends client with retry capabilities.
        
        Args:
            hl: Host language for Google Trends
            tz: Timezone offset in minutes from UTC
            retries: Number of retries for failed requests
            backoff_factor: Factor for exponential backoff between retries
        """
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.pytrends = TrendReq(
            hl=hl, 
            tz=tz,
            retries=retries,
            backoff_factor=backoff_factor,
            requests_args={'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}}
        )
    
    def get_interest_over_time(
        self,
        keyword: str,
        timeframe: str = 'today 3-m',
        geo: str = ''
    ) -> pd.DataFrame:
        """
        Get interest over time for a keyword.
        
        Args:
            keyword: Search term (e.g., "cargo pants")
            timeframe: Time range. Options:
                       - 'today 3-m' (last 3 months)
                       - 'today 12-m' (last 12 months)
                       - 'YYYY-MM-DD YYYY-MM-DD' (custom range)
            geo: Geographic region (e.g., 'US', 'GB', '' for worldwide)
            
        Returns:
            DataFrame with 'date' and 'interest' columns
        """
        # Build payload
        self.pytrends.build_payload(
            kw_list=[keyword],
            cat=0,          # Category: 0 = all
            timeframe=timeframe,
            geo=geo,
            gprop=''        # Property: '' = web search
        )
        
        # Fetch data
        df = self.pytrends.interest_over_time()
        
        if df.empty:
            return pd.DataFrame(columns=['date', 'interest'])
        
        # Clean up DataFrame
        df = df.reset_index()
        df = df[['date', keyword]].rename(columns={keyword: 'interest'})
        
        return df
    
    def calculate_slope(self, df: pd.DataFrame) -> Dict:
        """
        Calculate the slope (rate of growth) for interest data.
        
        Uses linear regression to determine trend direction
        and strength over the time period.
        
        Args:
            df: DataFrame with 'date' and 'interest' columns
            
        Returns:
            Dictionary with slope metrics:
            - slope: Rate of change per week
            - r_squared: Goodness of fit (0-1)
            - p_value: Statistical significance
            - trend: 'rising', 'falling', or 'stable'
            - growth_percent: Percentage change start to end
        """
        if len(df) < 2:
            return {
                'slope': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'trend': 'insufficient_data',
                'growth_percent': 0.0
            }
        
        # Convert dates to numeric (days from start)
        df = df.copy()
        df['days'] = (df['date'] - df['date'].min()).dt.days
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['days'],
            df['interest']
        )
        
        # Calculate percentage growth
        start_val = df['interest'].iloc[0]
        end_val = df['interest'].iloc[-1]
        
        if start_val > 0:
            growth_percent = ((end_val - start_val) / start_val) * 100
        else:
            growth_percent = 0.0 if end_val == 0 else 100.0
        
        # Determine trend category
        # Slope is per day, convert to per week for interpretability
        weekly_slope = slope * 7
        
        if p_value > 0.05:
            trend = 'stable'  # Not statistically significant
        elif weekly_slope > 0.5:
            trend = 'rising'
        elif weekly_slope < -0.5:
            trend = 'falling'
        else:
            trend = 'stable'
        
        return {
            'slope': round(weekly_slope, 4),
            'r_squared': round(r_value ** 2, 4),
            'p_value': round(p_value, 4),
            'trend': trend,
            'growth_percent': round(growth_percent, 2)
        }
    
    def analyze_keyword(
        self,
        keyword: str,
        timeframe: str = 'today 3-m',
        geo: str = ''
    ) -> Dict:
        """
        Full analysis of a fashion keyword.
        
        Args:
            keyword: Search term to analyze
            timeframe: Time range for analysis
            geo: Geographic region
            
        Returns:
            Complete analysis with data and metrics
        """
        # Get raw data
        df = self.get_interest_over_time(keyword, timeframe, geo)
        
        if df.empty:
            return {
                'keyword': keyword,
                'status': 'no_data',
                'data': None,
                'metrics': None
            }
        
        # Calculate metrics
        metrics = self.calculate_slope(df)
        
        return {
            'keyword': keyword,
            'timeframe': timeframe,
            'geo': geo if geo else 'worldwide',
            'status': 'success',
            'data': df.to_dict('records'),
            'metrics': metrics,
            'summary': self._generate_summary(keyword, metrics)
        }
    
    def _generate_summary(self, keyword: str, metrics: Dict) -> str:
        """Generate human-readable trend summary."""
        trend = metrics['trend']
        growth = metrics['growth_percent']
        
        if trend == 'rising':
            return f"📈 '{keyword}' is RISING with {growth:+.1f}% growth"
        elif trend == 'falling':
            return f"📉 '{keyword}' is DECLINING with {growth:+.1f}% change"
        else:
            return f"➡️ '{keyword}' is STABLE with {growth:+.1f}% change"
    
    def compare_keywords(
        self,
        keywords: List[str],
        timeframe: str = 'today 3-m',
        geo: str = ''
    ) -> pd.DataFrame:
        """
        Compare multiple keywords and rank by growth.
        
        Args:
            keywords: List of keywords to compare
            timeframe: Time range for analysis
            geo: Geographic region
            
        Returns:
            DataFrame ranked by slope (growth rate)
        """
        results = []
        
        for kw in keywords:
            analysis = self.analyze_keyword(kw, timeframe, geo)
            
            if analysis['status'] == 'success':
                results.append({
                    'keyword': kw,
                    'trend': analysis['metrics']['trend'],
                    'slope': analysis['metrics']['slope'],
                    'growth_percent': analysis['metrics']['growth_percent'],
                    'r_squared': analysis['metrics']['r_squared']
                })
            
            # Rate limiting (Google may block rapid requests)
            time.sleep(1)
        
        df = pd.DataFrame(results)
        return df.sort_values('slope', ascending=False)


# Usage Example
if __name__ == "__main__":
    tracker = TrendTracker()
    
    # Example 1: Analyze single keyword
    print("=" * 60)
    print("ANALYZING: 'cargo pants'")
    print("=" * 60)
    
    result = tracker.analyze_keyword("cargo pants", timeframe='today 3-m')
    
    if result['status'] == 'success':
        print(f"\n{result['summary']}")
        print(f"\nDetailed Metrics:")
        print(f"  - Weekly slope: {result['metrics']['slope']}")
        print(f"  - R² value: {result['metrics']['r_squared']}")
        print(f"  - P-value: {result['metrics']['p_value']}")
        print(f"  - Total growth: {result['metrics']['growth_percent']}%")
    
    # Example 2: Compare multiple fashion trends
    print("\n" + "=" * 60)
    print("COMPARING FASHION KEYWORDS")
    print("=" * 60)
    
    fashion_keywords = [
        "cargo pants",
        "wide leg jeans",
        "cropped blazer",
        "oversized sweater",
        "leather jacket"
    ]
    
    comparison = tracker.compare_keywords(fashion_keywords)
    print("\nKeyword Rankings by Growth Rate:")
    print(comparison.to_string(index=False))
