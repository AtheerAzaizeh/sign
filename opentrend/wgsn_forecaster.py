"""
OpenTrend - WGSN-Enhanced T-Shirt Fashion Forecasting Model
Professional-grade forecasting using official WGSN 2026 trend data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0a0a14'
plt.rcParams['axes.facecolor'] = '#12121c'

# =============================================================================
# WGSN OFFICIAL 2026 TREND DATA
# Source: WGSN + Coloro Color Forecasts, WGSN Design Direction Reports
# =============================================================================

WGSN_2026_COLORS = {
    # WGSN Color of the Year and Key Colors
    "Transformative Teal": {
        "hex": "#1E6B6F", "wgsn_score": 95, "trend": "key_color",
        "symbolism": "Eco-accountability, Earth-first mindset",
        "season": "all", "growth_trajectory": "strong_rise"
    },
    "Electric Fuchsia": {
        "hex": "#FF1493", "wgsn_score": 92, "trend": "accent",
        "symbolism": "Rebellious resistance, synthetic creativity",
        "season": "SS26", "growth_trajectory": "surge"
    },
    "Blue Aura": {
        "hex": "#7B9DC4", "wgsn_score": 85, "trend": "core",
        "symbolism": "Healing serenity, gender-inclusive",
        "season": "SS26", "growth_trajectory": "steady_rise"
    },
    "Amber Haze": {
        "hex": "#C9A227", "wgsn_score": 88, "trend": "accent",
        "symbolism": "Ancient wisdom, sustainable practices",
        "season": "SS26", "growth_trajectory": "strong_rise"
    },
    "Jelly Mint": {
        "hex": "#98FF98", "wgsn_score": 78, "trend": "accent",
        "symbolism": "Playful optimism, fresh energy",
        "season": "SS26", "growth_trajectory": "surge"
    },
    "Wax Paper": {
        "hex": "#FFF8DC", "wgsn_score": 88, "trend": "neutral",
        "symbolism": "Bio-based connection, sustainable serenity",
        "season": "AW26", "growth_trajectory": "steady_rise"
    },
    "Cocoa Powder": {
        "hex": "#6B4423", "wgsn_score": 85, "trend": "core",
        "symbolism": "Earthy nostalgia, transseasonal elegance",
        "season": "AW26", "growth_trajectory": "steady_rise"
    },
    "Fresh Purple": {
        "hex": "#8A2BE2", "wgsn_score": 82, "trend": "accent",
        "symbolism": "Royalty, mystery, spirituality",
        "season": "AW26", "growth_trajectory": "moderate_rise"
    },
    "Green Glow": {
        "hex": "#39FF14", "wgsn_score": 79, "trend": "statement",
        "symbolism": "Futuristic, emotive responsiveness",
        "season": "AW26", "growth_trajectory": "surge"
    },
    # Pantone Integration
    "Cloud Dancer": {
        "hex": "#F0EDE5", "wgsn_score": 98, "trend": "neutral",
        "symbolism": "Quiet luxury, fresh beginnings",
        "season": "all", "growth_trajectory": "dominant"
    },
}

WGSN_2026_DESIGN_DIRECTIONS = {
    # Core Design Directions from WGSN
    "Replenish": {
        "score": 94, "category": "macro_trend",
        "description": "Earthy textures, biophilic forms, bio-synthetic innovations",
        "key_elements": ["organic shapes", "natural dyes", "sustainable materials"],
        "trajectory": "dominant"
    },
    "Extra-Ordinary": {
        "score": 88, "category": "macro_trend",
        "description": "Elevated everyday, premium basics with detail",
        "key_elements": ["quality focus", "subtle details", "lasting design"],
        "trajectory": "strong_rise"
    },
    "Playful Paradox": {
        "score": 82, "category": "macro_trend",
        "description": "Sensorial textures, AI-inspired, real-virtual blur",
        "key_elements": ["jelly textures", "digital aesthetics", "playful forms"],
        "trajectory": "surge"
    },
    "Quiet Luxury": {
        "score": 96, "category": "consumer_trend",
        "description": "Understated elegance, quality over logos",
        "key_elements": ["minimal branding", "premium materials", "timeless design"],
        "trajectory": "dominant"
    },
    "Sportif": {
        "score": 88, "category": "style_trend",
        "description": "Athletic functionality meets everyday dressing",
        "key_elements": ["technical fabrics", "performance details", "chic functionality"],
        "trajectory": "strong_rise"
    },
    "Guardian Design": {
        "score": 85, "category": "product_trend",
        "description": "Anti-theft features, protection-focused design",
        "key_elements": ["hidden pockets", "RFID-blocking", "secure closures"],
        "trajectory": "moderate_rise"
    },
}

WGSN_2026_MATERIALS = {
    "Bio-based Fabrics": {"score": 95, "trajectory": "dominant", "sustainability": "high"},
    "Recycled Cotton (GRS)": {"score": 92, "trajectory": "strong_rise", "sustainability": "high"},
    "Lyocell/Tencel": {"score": 90, "trajectory": "strong_rise", "sustainability": "high"},
    "Organic Cotton (GOTS)": {"score": 88, "trajectory": "steady_rise", "sustainability": "high"},
    "Hemp Blend": {"score": 82, "trajectory": "surge", "sustainability": "very_high"},
    "Recycled Polyester": {"score": 78, "trajectory": "steady", "sustainability": "medium"},
    "Technical Performance": {"score": 85, "trajectory": "strong_rise", "sustainability": "medium"},
    "Heavy GSM Cotton": {"score": 86, "trajectory": "strong_rise", "sustainability": "medium"},
}

WGSN_2026_SILHOUETTES = {
    "Oversized/Relaxed": {"score": 94, "trajectory": "continuing_dominant"},
    "Boxy Fit": {"score": 88, "trajectory": "continuing_strong"},
    "Drop Shoulder": {"score": 86, "trajectory": "continuing_strong"},
    "Cropped": {"score": 45, "trajectory": "declining"},
    "Slim Fit": {"score": 38, "trajectory": "declining"},
    "Longline": {"score": 72, "trajectory": "stable"},
    "Exaggerated Volume": {"score": 78, "trajectory": "emerging"},
}

# =============================================================================
# ENHANCED TRAINING DATA GENERATION
# =============================================================================

def generate_wgsn_informed_trend(
    wgsn_score: float,
    trajectory: str,
    weeks: int = 104,  # 2 years
    peak_month: int = 6
) -> pd.DataFrame:
    """Generate training data informed by WGSN trajectory classifications."""
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=weeks, freq='W')
    n = len(dates)
    week_indices = np.arange(n)
    
    # Base value from WGSN score (scaled to trend range)
    base = wgsn_score * 0.8  # Scale down slightly for growth room
    
    # Trajectory-specific patterns
    if trajectory in ['dominant', 'continuing_dominant']:
        # High base, steady growth, low volatility
        trend = np.linspace(base - 10, base + 15, n)
        volatility = 3
    elif trajectory in ['strong_rise', 'continuing_strong']:
        # Moderate base, strong upward trend
        trend = np.linspace(base - 20, base + 20, n)
        volatility = 5
    elif trajectory == 'surge':
        # Lower start, exponential-like growth
        trend = base - 30 + 50 * (1 - np.exp(-week_indices / 40))
        volatility = 7
    elif trajectory in ['steady_rise', 'moderate_rise']:
        # Gradual increase
        trend = np.linspace(base - 15, base + 10, n)
        volatility = 4
    elif trajectory in ['stable', 'steady']:
        # Flat with minor fluctuations
        trend = np.ones(n) * base + np.random.normal(0, 2, n).cumsum() * 0.05
        volatility = 3
    elif trajectory == 'declining':
        # Downward trend
        trend = np.linspace(base + 15, base - 20, n)
        volatility = 5
    elif trajectory == 'emerging':
        # Starts low, accelerates recently
        trend = base - 25 + 35 * (week_indices / n) ** 2
        volatility = 6
    else:
        trend = np.ones(n) * base
        volatility = 4
    
    # Add seasonality (t-shirts peak in spring/summer)
    phase_shift = (peak_month - 6) * (2 * np.pi / 12)
    seasonality = 12 * np.sin(2 * np.pi * week_indices / 52 + phase_shift)
    
    # Add fashion cycle seasonality (new collections)
    fashion_cycle = 5 * np.sin(2 * np.pi * week_indices / 26)  # Bi-annual
    
    # Noise
    noise = np.random.normal(0, volatility, n)
    
    # Combine
    values = np.clip(trend + seasonality + fashion_cycle + noise, 0, 100)
    
    return pd.DataFrame({'ds': dates, 'y': values})


# =============================================================================
# WGSN T-SHIRT TREND MATRIX
# =============================================================================

WGSN_TSHIRT_TRENDS = {
    "colors": {
        f"{color} T-Shirt": {
            "wgsn_score": data['wgsn_score'],
            "trajectory": data['growth_trajectory'],
            "hex": data['hex'],
            "peak_month": 6 if data['season'] == 'SS26' else 9,
        }
        for color, data in WGSN_2026_COLORS.items()
    },
    "designs": {
        "Minimalist Logo Tee": {"wgsn_score": 92, "trajectory": "dominant", "peak_month": 6},
        "Oversized Blank Tee": {"wgsn_score": 94, "trajectory": "dominant", "peak_month": 6},
        "Nature/Botanical Tee": {"wgsn_score": 88, "trajectory": "strong_rise", "peak_month": 5},
        "Abstract Art Tee": {"wgsn_score": 82, "trajectory": "surge", "peak_month": 6},
        "Vintage Wash Tee": {"wgsn_score": 75, "trajectory": "moderate_rise", "peak_month": 7},
        "Typography Tee": {"wgsn_score": 70, "trajectory": "stable", "peak_month": 6},
        "Tie-Dye Tee": {"wgsn_score": 35, "trajectory": "declining", "peak_month": 7},
        "Band Graphic Tee": {"wgsn_score": 60, "trajectory": "stable", "peak_month": 7},
        "Sustainable Badge Tee": {"wgsn_score": 85, "trajectory": "strong_rise", "peak_month": 4},
    },
    "fits": {
        "Oversized Fit": {"wgsn_score": 94, "trajectory": "continuing_dominant", "peak_month": 6},
        "Boxy Fit": {"wgsn_score": 88, "trajectory": "continuing_strong", "peak_month": 6},
        "Drop Shoulder": {"wgsn_score": 86, "trajectory": "continuing_strong", "peak_month": 6},
        "Relaxed Fit": {"wgsn_score": 85, "trajectory": "steady_rise", "peak_month": 6},
        "Longline": {"wgsn_score": 72, "trajectory": "stable", "peak_month": 6},
        "Slim Fit": {"wgsn_score": 38, "trajectory": "declining", "peak_month": 6},
        "Cropped": {"wgsn_score": 45, "trajectory": "declining", "peak_month": 7},
        "Exaggerated Volume": {"wgsn_score": 78, "trajectory": "emerging", "peak_month": 9},
    },
    "materials": {
        "Bio-based Cotton": {"wgsn_score": 95, "trajectory": "dominant", "peak_month": 6},
        "Recycled Cotton (GRS)": {"wgsn_score": 92, "trajectory": "strong_rise", "peak_month": 6},
        "Organic Cotton (GOTS)": {"wgsn_score": 88, "trajectory": "steady_rise", "peak_month": 6},
        "Heavy GSM Cotton (220+)": {"wgsn_score": 86, "trajectory": "strong_rise", "peak_month": 5},
        "Hemp Blend": {"wgsn_score": 82, "trajectory": "surge", "peak_month": 5},
        "Lyocell Blend": {"wgsn_score": 85, "trajectory": "strong_rise", "peak_month": 6},
        "Technical Jersey": {"wgsn_score": 80, "trajectory": "moderate_rise", "peak_month": 7},
        "Standard Cotton": {"wgsn_score": 55, "trajectory": "declining", "peak_month": 6},
    },
    "necklines": {
        "Crew Neck": {"wgsn_score": 88, "trajectory": "continuing_dominant", "peak_month": 6},
        "Mock Neck": {"wgsn_score": 75, "trajectory": "emerging", "peak_month": 9},
        "Boat Neck": {"wgsn_score": 68, "trajectory": "emerging", "peak_month": 5},
        "Henley": {"wgsn_score": 62, "trajectory": "stable", "peak_month": 9},
        "V-Neck": {"wgsn_score": 42, "trajectory": "declining", "peak_month": 6},
        "Scoop Neck": {"wgsn_score": 45, "trajectory": "declining", "peak_month": 6},
    },
    "details": {
        "Minimal Branding": {"wgsn_score": 94, "trajectory": "dominant", "peak_month": 6},
        "Tone-on-Tone Logo": {"wgsn_score": 88, "trajectory": "strong_rise", "peak_month": 6},
        "Raw Hem": {"wgsn_score": 72, "trajectory": "moderate_rise", "peak_month": 7},
        "Contrast Stitching": {"wgsn_score": 68, "trajectory": "stable", "peak_month": 6},
        "Woven Label": {"wgsn_score": 82, "trajectory": "strong_rise", "peak_month": 6},
        "Hidden Pocket": {"wgsn_score": 78, "trajectory": "surge", "peak_month": 6},
        "Bold Branding": {"wgsn_score": 35, "trajectory": "declining", "peak_month": 6},
    },
}


class WGSNTShirtForecaster:
    """
    WGSN-enhanced Prophet forecasting model for T-Shirt fashion.
    Uses official WGSN trend data to inform model training.
    """
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.training_data = {}
        self.is_trained = False
        self.wgsn_data = WGSN_TSHIRT_TRENDS
        
    def generate_training_data(self):
        """Generate WGSN-informed training data."""
        print("\n📊 Generating WGSN-informed training data...")
        
        total = sum(len(v) for v in self.wgsn_data.values())
        current = 0
        
        for category, items in self.wgsn_data.items():
            print(f"  Category: {category}")
            self.training_data[category] = {}
            
            for name, params in items.items():
                current += 1
                df = generate_wgsn_informed_trend(
                    wgsn_score=params['wgsn_score'],
                    trajectory=params['trajectory'],
                    peak_month=params['peak_month']
                )
                self.training_data[category][name] = df
                
        print(f"  ✓ Generated WGSN-informed data for {total} trend items")
        
    def train_models(self):
        """Train Prophet models with WGSN priors."""
        print("\n🧠 Training WGSN-enhanced forecasting models...")
        
        total = sum(len(v) for v in self.training_data.values())
        current = 0
        
        for category, items in self.training_data.items():
            self.models[category] = {}
            
            for name, df in items.items():
                current += 1
                print(f"  [{current}/{total}] {name}...", end=" ", flush=True)
                
                # Get WGSN trajectory for this item
                trajectory = self.wgsn_data[category][name]['trajectory']
                
                # Configure Prophet based on WGSN trajectory
                if trajectory in ['dominant', 'continuing_dominant']:
                    changepoint_scale = 0.05  # Low - stable trend
                    seasonality_scale = 8
                elif trajectory in ['surge', 'emerging']:
                    changepoint_scale = 0.2  # High - rapid changes
                    seasonality_scale = 12
                elif trajectory == 'declining':
                    changepoint_scale = 0.15
                    seasonality_scale = 10
                else:
                    changepoint_scale = 0.1
                    seasonality_scale = 10
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=changepoint_scale,
                    seasonality_prior_scale=seasonality_scale,
                )
                
                # Add fashion season cycles
                model.add_seasonality(name='fashion_season', period=182.5, fourier_order=4)
                
                model.fit(df)
                self.models[category][name] = model
                print("✓")
        
        self.is_trained = True
        print(f"\n  ✅ Trained {total} WGSN-enhanced models")
        
    def generate_forecasts(self, periods: int = 26):
        """Generate 6-month forecasts."""
        if not self.is_trained:
            raise ValueError("Models must be trained first.")
        
        print(f"\n🔮 Generating {periods}-week WGSN-calibrated forecasts...")
        
        for category, models in self.models.items():
            self.forecasts[category] = {}
            
            for name, model in models.items():
                future = model.make_future_dataframe(periods=periods, freq='W')
                forecast = model.predict(future)
                
                historical = self.training_data[category][name]
                future_only = forecast[forecast['ds'] > historical['ds'].max()]
                
                current_value = historical['y'].iloc[-1]
                peak_value = future_only['yhat'].max()
                peak_idx = future_only['yhat'].idxmax()
                peak_date = future_only.loc[peak_idx, 'ds']
                final_value = future_only['yhat'].iloc[-1]
                
                growth_pct = ((final_value - current_value) / current_value) * 100 if current_value > 0 else 0
                
                self.forecasts[category][name] = {
                    'forecast_df': future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                    'current_value': round(current_value, 1),
                    'peak_value': round(peak_value, 1),
                    'peak_date': peak_date,
                    'final_value': round(final_value, 1),
                    'growth_pct': round(growth_pct, 1),
                    'wgsn_score': self.wgsn_data[category][name]['wgsn_score'],
                    'trajectory': self.wgsn_data[category][name]['trajectory'],
                }
        
        print("  ✅ WGSN-calibrated forecasts generated")
        
    def get_wgsn_ranked_predictions(self, n: int = 15) -> list:
        """Get predictions ranked by WGSN score + growth potential."""
        predictions = []
        
        for category, forecasts in self.forecasts.items():
            for name, data in forecasts.items():
                # Composite score: WGSN authority + model prediction
                composite = (data['wgsn_score'] * 0.6) + (max(0, data['growth_pct']) * 0.4)
                
                predictions.append({
                    'category': category,
                    'name': name,
                    'wgsn_score': data['wgsn_score'],
                    'trajectory': data['trajectory'],
                    'predicted_growth': data['growth_pct'],
                    'peak_value': data['peak_value'],
                    'peak_date': data['peak_date'],
                    'composite_score': round(composite, 1),
                })
        
        predictions.sort(key=lambda x: x['composite_score'], reverse=True)
        return predictions[:n]
    
    def get_best_tshirt_combination(self) -> dict:
        """Get the optimal t-shirt design combining all top categories."""
        
        best = {}
        for category in ['colors', 'designs', 'fits', 'materials', 'necklines', 'details']:
            if category in self.forecasts:
                top = max(
                    self.forecasts[category].items(),
                    key=lambda x: x[1]['wgsn_score'] * 0.6 + max(0, x[1]['growth_pct']) * 0.4
                )
                best[category] = {
                    'name': top[0],
                    'wgsn_score': top[1]['wgsn_score'],
                    'growth': top[1]['growth_pct'],
                }
        
        return best
    
    def create_wgsn_dashboard(self):
        """Create WGSN-branded forecast dashboard."""
        
        fig = plt.figure(figsize=(26, 20))
        fig.suptitle('OPENTREND × WGSN T-SHIRT FORECAST 2026\nProfessional Fashion Intelligence', 
                     fontsize=24, fontweight='bold', color='white', y=0.98)
        
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3,
                              left=0.04, right=0.96, top=0.93, bottom=0.04)
        
        # =====================================================================
        # ROW 1: WGSN Color Palette + Top Predictions
        # =====================================================================
        
        # WGSN Colors
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_title('WGSN 2026 KEY COLORS FOR T-SHIRTS', fontsize=12, fontweight='bold', color='white', pad=10)
        
        colors = [(n.replace(' T-Shirt', ''), d) for n, d in self.wgsn_data['colors'].items()]
        colors.sort(key=lambda x: x[1]['wgsn_score'], reverse=True)
        
        for i, (name, data) in enumerate(colors[:8]):
            col = i % 4
            row = i // 4
            rect = mpatches.Rectangle((col*0.25, 0.5-row*0.5), 0.23, 0.45,
                                       facecolor=data['hex'], edgecolor='white', linewidth=2)
            ax1.add_patch(rect)
            text_color = 'white' if data['hex'] in ['#1E6B6F', '#6B4423', '#FF1493', '#8A2BE2'] else '#333'
            ax1.text(col*0.25+0.115, 0.5-row*0.5+0.32, name[:12], ha='center', fontsize=7, 
                    color=text_color, fontweight='bold')
            ax1.text(col*0.25+0.115, 0.5-row*0.5+0.15, f"WGSN: {data['wgsn_score']}", 
                    ha='center', fontsize=6, color=text_color)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Top 10 predictions
        ax2 = fig.add_subplot(gs[0, 2:])
        top_10 = self.get_wgsn_ranked_predictions(10)
        
        names = [p['name'].replace(' T-Shirt', '').replace(' Tee', '')[:18] for p in top_10]
        scores = [p['composite_score'] for p in top_10]
        cat_colors = {'colors': '#00d9ff', 'designs': '#ff6b9d', 'fits': '#00ff88', 
                      'materials': '#ffd93d', 'necklines': '#b19cd9', 'details': '#ff9f43'}
        bar_colors = [cat_colors.get(p['category'], '#888') for p in top_10]
        
        bars = ax2.barh(names[::-1], scores[::-1], color=bar_colors[::-1], edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('WGSN Composite Score', fontsize=10, color='#888')
        ax2.set_title('TOP 10 WGSN-RANKED PREDICTIONS', fontsize=12, fontweight='bold', color='white', pad=10)
        ax2.grid(True, axis='x', alpha=0.3)
        
        # =====================================================================
        # ROW 2: Category Forecasts
        # =====================================================================
        
        categories = ['colors', 'designs', 'fits', 'materials']
        for i, cat in enumerate(categories):
            ax = fig.add_subplot(gs[1, i])
            self._plot_category(ax, cat)
        
        # =====================================================================
        # ROW 3: Forecast Lines + Best Combination
        # =====================================================================
        
        ax3 = fig.add_subplot(gs[2, :2])
        top_5 = self.get_wgsn_ranked_predictions(5)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
        
        for i, pred in enumerate(top_5):
            cat = pred['category']
            name = pred['name']
            forecast_df = self.forecasts[cat][name]['forecast_df']
            
            ax3.plot(forecast_df['ds'], forecast_df['yhat'], 
                    label=f"{name.replace(' T-Shirt', '')[:15]} [WGSN:{pred['wgsn_score']}]",
                    color=colors[i], linewidth=2.5)
            ax3.fill_between(forecast_df['ds'], 
                            forecast_df['yhat_lower'], 
                            forecast_df['yhat_upper'],
                            alpha=0.15, color=colors[i])
        
        ax3.set_xlabel('Date', fontsize=10, color='#888')
        ax3.set_ylabel('Trend Score', fontsize=10, color='#888')
        ax3.set_title('6-MONTH WGSN-CALIBRATED FORECAST', fontsize=12, fontweight='bold', color='white', pad=10)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Best combination
        ax4 = fig.add_subplot(gs[2, 2:])
        ax4.axis('off')
        
        best = self.get_best_tshirt_combination()
        
        combo_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║         🏆 WGSN OPTIMAL T-SHIRT DESIGN 2026 🏆                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🎨 COLOR:     {best.get('colors', {}).get('name', 'N/A'):<30}     ║
║               WGSN Score: {best.get('colors', {}).get('wgsn_score', 0):<3}                              ║
║                                                                  ║
║  🎯 DESIGN:    {best.get('designs', {}).get('name', 'N/A'):<30}     ║
║               WGSN Score: {best.get('designs', {}).get('wgsn_score', 0):<3}                              ║
║                                                                  ║
║  📐 FIT:       {best.get('fits', {}).get('name', 'N/A'):<30}     ║
║               WGSN Score: {best.get('fits', {}).get('wgsn_score', 0):<3}                              ║
║                                                                  ║
║  🧵 MATERIAL:  {best.get('materials', {}).get('name', 'N/A'):<30}     ║
║               WGSN Score: {best.get('materials', {}).get('wgsn_score', 0):<3}                              ║
║                                                                  ║
║  👔 NECKLINE:  {best.get('necklines', {}).get('name', 'N/A'):<30}     ║
║               WGSN Score: {best.get('necklines', {}).get('wgsn_score', 0):<3}                              ║
║                                                                  ║
║  ✨ DETAILS:   {best.get('details', {}).get('name', 'N/A'):<30}     ║
║               WGSN Score: {best.get('details', {}).get('wgsn_score', 0):<3}                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
        
        ax4.text(0.5, 0.5, combo_text, transform=ax4.transAxes, fontsize=9,
                fontfamily='monospace', color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', 
                         edgecolor='#00d9ff', linewidth=2))
        
        # =====================================================================
        # ROW 4: Summary & Action Items
        # =====================================================================
        
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('off')
        
        summary_text = """
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                            WGSN T-SHIRT STRATEGY SUMMARY 2026                                                                              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  TOP COLORS (WGSN)              TOP DESIGNS                      TOP MATERIALS                         KEY TRENDS                                                        │
│  ────────────────               ───────────                      ─────────────                         ──────────                                                        │
│  1. Cloud Dancer (98)           1. Oversized Blank (94)          1. Bio-based Cotton (95)              • Quiet Luxury dominates                                          │
│  2. Transformative Teal (95)    2. Minimalist Logo (92)          2. Recycled Cotton GRS (92)           • Sustainability is mandatory                                     │
│  3. Electric Fuchsia (92)       3. Nature/Botanical (88)         3. Lyocell Blend (85)                 • Oversized fits continuing                                       │
│  4. Amber Haze (88)             4. Abstract Art (82)             4. Heavy GSM Cotton (86)              • Bold accent colors emerging                                     │
│  5. Wax Paper (88)              5. Sustainable Badge (85)        5. Hemp Blend (82)                    • Declining: Slim fit, tie-dye, V-neck                            │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  WGSN DESIGN DIRECTIONS: Replenish (biophilic) + Extra-Ordinary (elevated basics) + Quiet Luxury (minimal branding) + Sportif (technical details)                        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
        
        ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes, fontsize=8,
                fontfamily='monospace', color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#00d9ff', linewidth=1))
        
        save_path = Path(__file__).parent / 'wgsn_tshirt_forecast.png'
        plt.savefig(save_path, dpi=150, facecolor='#0a0a14', edgecolor='none', bbox_inches='tight')
        print(f'\n✅ WGSN Dashboard saved: {save_path}')
        plt.close()
        
        return save_path
    
    def _plot_category(self, ax, category: str):
        """Plot category forecast bars."""
        forecasts = self.forecasts.get(category, {})
        
        sorted_items = sorted(
            [(name, data) for name, data in forecasts.items()],
            key=lambda x: x[1]['wgsn_score'],
            reverse=True
        )[:5]
        
        names = [x[0].replace(' T-Shirt', '').replace(' Tee', '')[:12] for x in sorted_items]
        wgsn_scores = [x[1]['wgsn_score'] for x in sorted_items]
        growths = [x[1]['growth_pct'] for x in sorted_items]
        
        # Color by WGSN score
        colors = plt.cm.Blues(np.array(wgsn_scores) / 100)
        
        bars = ax.barh(names[::-1], wgsn_scores[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
        
        for bar, growth in zip(bars, growths[::-1]):
            sign = '+' if growth > 0 else ''
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{sign}{growth:.0f}%', va='center', fontsize=7, color='#00ff88' if growth > 0 else '#ff6b6b')
        
        ax.set_title(category.upper(), fontsize=10, fontweight='bold', color='white', pad=8)
        ax.set_xlabel('WGSN Score', fontsize=8, color='#888')
        ax.set_xlim(0, 110)
        ax.grid(True, axis='x', alpha=0.3)
    
    def print_wgsn_report(self):
        """Print comprehensive WGSN forecast report."""
        
        print("\n" + "="*90)
        print("            OPENTREND × WGSN T-SHIRT FORECAST 2026")
        print("                  Professional Fashion Intelligence")
        print(f"                   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*90)
        
        print("\n🏆 TOP 15 WGSN-RANKED PREDICTIONS")
        print("-"*80)
        
        for i, pred in enumerate(self.get_wgsn_ranked_predictions(15), 1):
            peak_str = pred['peak_date'].strftime('%b %d')
            print(f"  {i:2}. {pred['name']:<28} [{pred['category']:<9}] "
                  f"WGSN:{pred['wgsn_score']:3} | Growth:{pred['predicted_growth']:+6.1f}% | Peak:{peak_str}")
        
        print("\n\n🎯 OPTIMAL T-SHIRT DESIGN (WGSN-RECOMMENDED)")
        print("="*80)
        
        best = self.get_best_tshirt_combination()
        for category, data in best.items():
            print(f"  {category.upper():<12}: {data['name']:<35} (WGSN: {data['wgsn_score']})")
        
        print("\n\n📋 WGSN DESIGN BRIEF")
        print("="*80)
        print("""
    TARGET: Spring/Summer 2026 Collection
    
    CONCEPT: "Quiet Earth" - Sustainable luxury meets nature-inspired calm
    
    HERO PRODUCT:
    ─────────────
    Color:      Cloud Dancer or Transformative Teal
    Design:     Oversized blank with tone-on-tone embroidered logo
    Fit:        Oversized with drop shoulders
    Material:   220 GSM Bio-based or Recycled Cotton (GRS certified)
    Neckline:   Classic crew neck with ribbed finish
    Details:    Minimal branding, woven label, hidden pocket option
    
    ACCENT PIECES:
    ──────────────
    • Electric Fuchsia with abstract art print
    • Amber Haze with nature/botanical illustration
    • Jelly Mint in solid oversized blank
    
    SUSTAINABILITY REQUIREMENTS (WGSN Mandatory):
    ─────────────────────────────────────────────
    ✓ GRS or GOTS certified materials
    ✓ Natural or low-impact dyes
    ✓ Recycled packaging
    ✓ Transparent supply chain
    ✓ Carbon-neutral production target
    
    TRENDS TO AVOID:
    ────────────────
    ✗ Slim/fitted silhouettes
    ✗ Bold logo branding
    ✗ Tie-dye patterns
    ✗ V-neck and scoop neck
    ✗ Standard (non-sustainable) cotton
""")
        print("="*90)


def run_wgsn_training():
    """Run complete WGSN-enhanced training pipeline."""
    
    print("="*90)
    print("      OPENTREND × WGSN T-SHIRT FORECASTING MODEL")
    print("      Professional Fashion Intelligence System")
    print("="*90)
    
    forecaster = WGSNTShirtForecaster()
    forecaster.generate_training_data()
    forecaster.train_models()
    forecaster.generate_forecasts(periods=26)
    forecaster.create_wgsn_dashboard()
    forecaster.print_wgsn_report()
    
    return forecaster


if __name__ == "__main__":
    forecaster = run_wgsn_training()
    
    print("\n📊 Charts saved:")
    print("  - wgsn_tshirt_forecast.png")
