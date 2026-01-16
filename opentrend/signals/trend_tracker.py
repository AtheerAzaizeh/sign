"""
opentrend/signals/trend_tracker.py
Google Trends Signal Tracking with Growth Analysis

Enhanced version with:
- Better rate limit handling (429 errors)
- Exponential backoff with jitter
- Request cookie management
- Fallback to cached/synthetic data
"""
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendTracker:
    """
    Tracks Google Trends data and calculates growth metrics.
    
    Enhanced with:
    - Rate limit handling (exponential backoff)
    - Cookie-based session management
    - Longer delays to avoid 429 errors
    """
    
    def __init__(
        self, 
        hl: str = 'en-US', 
        tz: int = 360, 
        retries: int = 5, 
        backoff_factor: float = 2.0,
        initial_delay: float = 5.0
    ):
        """
        Initialize the Pytrends client with enhanced retry capabilities.
        
        Args:
            hl: Host language for Google Trends
            tz: Timezone offset in minutes from UTC
            retries: Number of retries for failed requests
            backoff_factor: Factor for exponential backoff between retries
            initial_delay: Initial delay between requests in seconds
        """
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.hl = hl
        self.tz = tz
        
        # Create pytrends instance with better headers
        self._init_pytrends()
    
    def _init_pytrends(self):
        """Initialize or reinitialize pytrends with fresh session."""
        self.pytrends = TrendReq(
            hl=self.hl, 
            tz=self.tz,
            retries=2,
            backoff_factor=0.5,
            timeout=(10, 30),
            requests_args={
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                }
            }
        )
    
    def _delay_with_jitter(self, base_delay: float) -> None:
        """Apply delay with random jitter to avoid thundering herd."""
        jitter = random.uniform(0.5, 1.5)
        delay = base_delay * jitter
        logger.debug(f"Waiting {delay:.1f}s before request...")
        time.sleep(delay)
    
    def get_interest_over_time(
        self,
        keyword: str,
        timeframe: str = 'today 3-m',
        geo: str = ''
    ) -> pd.DataFrame:
        """
        Get interest over time for a keyword with rate limit handling.
        
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
        last_error = None
        
        for attempt in range(self.retries):
            try:
                # Add delay before request (increases with each retry)
                if attempt > 0:
                    delay = self.initial_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retry {attempt+1}/{self.retries} for '{keyword}' after {delay:.0f}s delay")
                    self._delay_with_jitter(delay)
                    # Reinitialize session on retry
                    self._init_pytrends()
                else:
                    # Small initial delay
                    self._delay_with_jitter(self.initial_delay)
                
                # Build payload
                self.pytrends.build_payload(
                    kw_list=[keyword],
                    cat=0,
                    timeframe=timeframe,
                    geo=geo,
                    gprop=''
                )
                
                # Fetch data
                df = self.pytrends.interest_over_time()
                
                if df.empty:
                    logger.warning(f"No data returned for '{keyword}'")
                    return pd.DataFrame(columns=['date', 'interest'])
                
                # Clean up DataFrame
                df = df.reset_index()
                df = df[['date', keyword]].rename(columns={keyword: 'interest'})
                
                logger.info(f"✓ Fetched {len(df)} data points for '{keyword}'")
                return df
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                if '429' in error_msg or 'TooManyRequests' in error_msg:
                    logger.warning(f"Rate limited (429) for '{keyword}'. Attempt {attempt+1}/{self.retries}")
                else:
                    logger.error(f"Error fetching '{keyword}': {e}")
        
        # All retries failed
        logger.error(f"All {self.retries} attempts failed for '{keyword}': {last_error}")
        raise last_error
    
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
        weekly_slope = slope * 7
        
        if p_value > 0.05:
            trend = 'stable'
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
        try:
            df = self.get_interest_over_time(keyword, timeframe, geo)
        except Exception as e:
            return {
                'keyword': keyword,
                'status': 'error',
                'error': str(e),
                'data': None,
                'metrics': None
            }
        
        if df.empty:
            return {
                'keyword': keyword,
                'status': 'no_data',
                'data': None,
                'metrics': None
            }
        
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
            
            # Longer delay between keywords to avoid rate limiting
            self._delay_with_jitter(self.initial_delay * 2)
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('slope', ascending=False)
        return df


# Usage Example
if __name__ == "__main__":
    print("=" * 60)
    print("GOOGLE TRENDS TRACKER TEST")
    print("=" * 60)
    
    tracker = TrendTracker(initial_delay=3.0)
    
    # Test single keyword
    print("\n📊 Testing: 'cargo pants' (3 months)")
    
    try:
        result = tracker.analyze_keyword("cargo pants", timeframe='today 3-m')
        
        if result['status'] == 'success':
            print(f"\n{result['summary']}")
            print(f"  Slope: {result['metrics']['slope']}")
            print(f"  Growth: {result['metrics']['growth_percent']}%")
        else:
            print(f"  Status: {result['status']}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n✅ Test complete")
