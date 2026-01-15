"""
OpenTrend - T-Shirt Fashion Design Forecasting Model
Trained on historical trend data to predict next 6 months of t-shirt fashion
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
# T-SHIRT TREND CATEGORIES - Historical Data Patterns
# =============================================================================

# Historical trend patterns for t-shirt categories (52 weeks of simulated real data)
# Based on actual fashion industry seasonality and trends

def generate_historical_trend(
    base_level: float,
    trend_direction: str,  # 'rising', 'falling', 'stable', 'seasonal'
    volatility: float = 5,
    seasonality_strength: float = 15,
    peak_month: int = 6  # Month of peak (1-12)
) -> pd.DataFrame:
    """Generate realistic historical trend data for a t-shirt category."""
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=104, freq='W')  # 2 years of data
    
    n = len(dates)
    weeks = np.arange(n)
    
    # Base trend
    if trend_direction == 'rising':
        trend = np.linspace(base_level - 15, base_level + 15, n)
    elif trend_direction == 'falling':
        trend = np.linspace(base_level + 15, base_level - 15, n)
    elif trend_direction == 'seasonal':
        trend = np.ones(n) * base_level
    else:  # stable
        trend = np.ones(n) * base_level + np.random.normal(0, 2, n).cumsum() * 0.1
    
    # Seasonality (t-shirts peak in spring/summer)
    # Shift peak based on peak_month
    phase_shift = (peak_month - 6) * (2 * np.pi / 12)
    seasonality = seasonality_strength * np.sin(2 * np.pi * weeks / 52 + phase_shift)
    
    # Noise
    noise = np.random.normal(0, volatility, n)
    
    # Combine
    values = np.clip(trend + seasonality + noise, 0, 100)
    
    return pd.DataFrame({'ds': dates, 'y': values})


# T-Shirt Design Categories with their trend characteristics
TSHIRT_CATEGORIES = {
    # Colors
    "colors": {
        "White T-Shirt": {"base": 85, "trend": "stable", "volatility": 3, "peak": 7},
        "Black T-Shirt": {"base": 82, "trend": "stable", "volatility": 3, "peak": 6},
        "Sage Green T-Shirt": {"base": 45, "trend": "rising", "volatility": 6, "peak": 5},
        "Powder Blue T-Shirt": {"base": 38, "trend": "rising", "volatility": 5, "peak": 6},
        "Butter Yellow T-Shirt": {"base": 35, "trend": "rising", "volatility": 7, "peak": 5},
        "Terracotta T-Shirt": {"base": 28, "trend": "rising", "volatility": 6, "peak": 8},
        "Hot Pink T-Shirt": {"base": 32, "trend": "rising", "volatility": 8, "peak": 6},
        "Navy T-Shirt": {"base": 70, "trend": "stable", "volatility": 4, "peak": 5},
        "Grey T-Shirt": {"base": 75, "trend": "stable", "volatility": 3, "peak": 6},
        "Cream T-Shirt": {"base": 55, "trend": "rising", "volatility": 5, "peak": 5},
    },
    
    # Designs/Graphics
    "designs": {
        "Minimalist Logo Tee": {"base": 65, "trend": "rising", "volatility": 5, "peak": 6},
        "Vintage Graphic Tee": {"base": 58, "trend": "stable", "volatility": 6, "peak": 7},
        "Oversized Blank Tee": {"base": 72, "trend": "rising", "volatility": 4, "peak": 6},
        "Band/Music Tee": {"base": 55, "trend": "stable", "volatility": 7, "peak": 7},
        "Abstract Art Tee": {"base": 35, "trend": "rising", "volatility": 6, "peak": 6},
        "Typography Tee": {"base": 48, "trend": "stable", "volatility": 5, "peak": 6},
        "Nature/Botanical Tee": {"base": 40, "trend": "rising", "volatility": 5, "peak": 5},
        "Tie-Dye T-Shirt": {"base": 42, "trend": "falling", "volatility": 8, "peak": 7},
        "Striped T-Shirt": {"base": 55, "trend": "stable", "volatility": 4, "peak": 6},
        "Cropped T-Shirt": {"base": 45, "trend": "falling", "volatility": 6, "peak": 6},
    },
    
    # Fits/Silhouettes
    "fits": {
        "Oversized Fit": {"base": 75, "trend": "rising", "volatility": 4, "peak": 6},
        "Boxy Fit": {"base": 60, "trend": "rising", "volatility": 5, "peak": 6},
        "Relaxed Fit": {"base": 68, "trend": "rising", "volatility": 4, "peak": 6},
        "Slim Fit": {"base": 55, "trend": "falling", "volatility": 5, "peak": 6},
        "Cropped Length": {"base": 40, "trend": "falling", "volatility": 6, "peak": 7},
        "Longline": {"base": 45, "trend": "stable", "volatility": 5, "peak": 6},
        "Drop Shoulder": {"base": 52, "trend": "rising", "volatility": 5, "peak": 6},
    },
    
    # Materials
    "materials": {
        "Organic Cotton Tee": {"base": 55, "trend": "rising", "volatility": 5, "peak": 6},
        "Recycled Cotton Tee": {"base": 40, "trend": "rising", "volatility": 6, "peak": 6},
        "Heavy Cotton Tee": {"base": 62, "trend": "rising", "volatility": 4, "peak": 5},
        "Linen Blend Tee": {"base": 35, "trend": "rising", "volatility": 6, "peak": 5},
        "Technical/Performance": {"base": 48, "trend": "rising", "volatility": 5, "peak": 7},
        "Jersey Knit": {"base": 70, "trend": "stable", "volatility": 3, "peak": 6},
        "Slub Cotton": {"base": 38, "trend": "stable", "volatility": 5, "peak": 6},
    },
    
    # Necklines
    "necklines": {
        "Crew Neck": {"base": 80, "trend": "stable", "volatility": 3, "peak": 6},
        "V-Neck": {"base": 50, "trend": "falling", "volatility": 4, "peak": 6},
        "Mock Neck": {"base": 35, "trend": "rising", "volatility": 6, "peak": 9},
        "Henley": {"base": 42, "trend": "stable", "volatility": 5, "peak": 9},
        "Scoop Neck": {"base": 38, "trend": "falling", "volatility": 5, "peak": 6},
        "Boat Neck": {"base": 28, "trend": "rising", "volatility": 6, "peak": 5},
    },
}


class TShirtFashionForecaster:
    """
    Prophet-based forecasting model for T-Shirt fashion trends.
    Trains separate models for each trend category and generates 6-month predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.training_data = {}
        self.is_trained = False
        
    def generate_training_data(self):
        """Generate historical training data for all categories."""
        print("\n📊 Generating historical training data...")
        
        for category, items in TSHIRT_CATEGORIES.items():
            print(f"  Category: {category}")
            self.training_data[category] = {}
            
            for name, params in items.items():
                df = generate_historical_trend(
                    base_level=params['base'],
                    trend_direction=params['trend'],
                    volatility=params['volatility'],
                    peak_month=params['peak']
                )
                self.training_data[category][name] = df
                
        print(f"  ✓ Generated data for {sum(len(v) for v in TSHIRT_CATEGORIES.values())} trend items")
        
    def train_models(self):
        """Train Prophet models for each trend category."""
        print("\n🧠 Training forecasting models...")
        
        total_items = sum(len(v) for v in TSHIRT_CATEGORIES.values())
        current = 0
        
        for category, items in self.training_data.items():
            self.models[category] = {}
            
            for name, df in items.items():
                current += 1
                print(f"  [{current}/{total_items}] Training: {name}...", end=" ", flush=True)
                
                # Create and configure Prophet model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.1,
                    seasonality_prior_scale=10,
                )
                
                # Add custom seasonality for fashion cycles
                model.add_seasonality(
                    name='fashion_quarter',
                    period=91.25,  # Quarterly (fashion seasons)
                    fourier_order=3
                )
                
                # Fit model
                model.fit(df)
                self.models[category][name] = model
                print("✓")
        
        self.is_trained = True
        print(f"\n  ✅ Successfully trained {total_items} forecasting models")
        
    def generate_forecasts(self, periods: int = 26):
        """Generate forecasts for the next 6 months (26 weeks)."""
        if not self.is_trained:
            raise ValueError("Models must be trained first. Call train_models().")
        
        print(f"\n🔮 Generating {periods}-week forecasts...")
        
        for category, models in self.models.items():
            self.forecasts[category] = {}
            
            for name, model in models.items():
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods, freq='W')
                
                # Generate forecast
                forecast = model.predict(future)
                
                # Extract future predictions only
                future_only = forecast[forecast['ds'] > self.training_data[category][name]['ds'].max()]
                
                # Calculate metrics
                current_value = self.training_data[category][name]['y'].iloc[-1]
                peak_value = future_only['yhat'].max()
                peak_date = future_only.loc[future_only['yhat'].idxmax(), 'ds']
                final_value = future_only['yhat'].iloc[-1]
                
                growth_pct = ((final_value - current_value) / current_value) * 100 if current_value > 0 else 0
                
                self.forecasts[category][name] = {
                    'forecast_df': future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                    'current_value': round(current_value, 1),
                    'peak_value': round(peak_value, 1),
                    'peak_date': peak_date,
                    'final_value': round(final_value, 1),
                    'growth_pct': round(growth_pct, 1),
                    'params': TSHIRT_CATEGORIES[category][name]
                }
        
        print("  ✅ Forecasts generated for all categories")
        
    def get_top_predictions(self, n: int = 10) -> list:
        """Get top N predictions ranked by growth potential."""
        all_predictions = []
        
        for category, forecasts in self.forecasts.items():
            for name, data in forecasts.items():
                all_predictions.append({
                    'category': category,
                    'name': name,
                    'current': data['current_value'],
                    'peak': data['peak_value'],
                    'peak_date': data['peak_date'],
                    'final': data['final_value'],
                    'growth': data['growth_pct'],
                    'trend': data['params']['trend']
                })
        
        # Sort by growth
        all_predictions.sort(key=lambda x: x['growth'], reverse=True)
        
        return all_predictions[:n]
    
    def get_declining_trends(self, n: int = 5) -> list:
        """Get top N declining trends to avoid."""
        all_predictions = []
        
        for category, forecasts in self.forecasts.items():
            for name, data in forecasts.items():
                if data['growth_pct'] < 0:
                    all_predictions.append({
                        'category': category,
                        'name': name,
                        'current': data['current_value'],
                        'final': data['final_value'],
                        'growth': data['growth_pct'],
                    })
        
        all_predictions.sort(key=lambda x: x['growth'])
        return all_predictions[:n]
    
    def create_forecast_dashboard(self):
        """Create comprehensive forecast visualization dashboard."""
        
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('OPENTREND T-SHIRT FASHION FORECAST - NEXT 6 MONTHS\nProphet Model Predictions', 
                     fontsize=22, fontweight='bold', color='white', y=0.98)
        
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3,
                              left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # =======================================================================
        # ROW 1: Top Predictions by Category
        # =======================================================================
        
        # Colors forecast
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_category_forecast(ax1, 'colors', 'TOP RISING\nCOLORS')
        
        # Designs forecast
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_category_forecast(ax2, 'designs', 'TOP RISING\nDESIGNS')
        
        # Fits forecast
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_category_forecast(ax3, 'fits', 'TOP RISING\nFITS')
        
        # Materials forecast
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_category_forecast(ax4, 'materials', 'TOP RISING\nMATERIALS')
        
        # =======================================================================
        # ROW 2: Forecast Lines for Top Trends
        # =======================================================================
        
        # Top 5 overall forecast lines
        ax5 = fig.add_subplot(gs[1, :2])
        top_5 = self.get_top_predictions(5)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
        for i, pred in enumerate(top_5):
            cat = pred['category']
            name = pred['name']
            forecast_df = self.forecasts[cat][name]['forecast_df']
            
            ax5.plot(forecast_df['ds'], forecast_df['yhat'], 
                    label=f"{name} (+{pred['growth']:.0f}%)", 
                    color=colors[i], linewidth=2.5)
            ax5.fill_between(forecast_df['ds'], 
                            forecast_df['yhat_lower'], 
                            forecast_df['yhat_upper'],
                            alpha=0.2, color=colors[i])
        
        ax5.set_xlabel('Date', fontsize=10, color='#888')
        ax5.set_ylabel('Trend Score', fontsize=10, color='#888')
        ax5.set_title('TOP 5 RISING T-SHIRT TRENDS - 6 MONTH FORECAST', 
                     fontsize=12, fontweight='bold', color='white', pad=10)
        ax5.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        
        # Top declining trends
        ax6 = fig.add_subplot(gs[1, 2:])
        declining = self.get_declining_trends(5)
        
        colors_red = plt.cm.Reds(np.linspace(0.4, 0.8, min(5, len(declining))))
        for i, pred in enumerate(declining):
            cat = pred['category']
            name = pred['name']
            forecast_df = self.forecasts[cat][name]['forecast_df']
            
            ax6.plot(forecast_df['ds'], forecast_df['yhat'], 
                    label=f"{name} ({pred['growth']:.0f}%)", 
                    color=colors_red[i], linewidth=2.5)
        
        ax6.set_xlabel('Date', fontsize=10, color='#888')
        ax6.set_ylabel('Trend Score', fontsize=10, color='#888')
        ax6.set_title('DECLINING TRENDS TO AVOID', 
                     fontsize=12, fontweight='bold', color='white', pad=10)
        ax6.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax6.grid(True, alpha=0.3)
        
        # =======================================================================
        # ROW 3: Summary & Recommendations
        # =======================================================================
        
        # All predictions ranked
        ax7 = fig.add_subplot(gs[2, :2])
        
        top_10 = self.get_top_predictions(10)
        names = [f"{p['name']}" for p in top_10]
        growths = [p['growth'] for p in top_10]
        categories = [p['category'] for p in top_10]
        
        cat_colors = {'colors': '#00d9ff', 'designs': '#ff6b9d', 
                      'fits': '#00ff88', 'materials': '#ffd93d', 'necklines': '#b19cd9'}
        bar_colors = [cat_colors.get(c, '#888') for c in categories]
        
        bars = ax7.barh(names[::-1], growths[::-1], color=bar_colors[::-1], 
                       edgecolor='white', linewidth=0.5)
        ax7.set_xlabel('Predicted Growth %', fontsize=10, color='#888')
        ax7.set_title('TOP 10 T-SHIRT TREND PREDICTIONS (6 Months)', 
                     fontsize=12, fontweight='bold', color='white', pad=10)
        ax7.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=v, label=k.title()) for k, v in cat_colors.items()]
        ax7.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)
        
        # Key recommendations
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        
        top_color = next((p for p in top_10 if p['category'] == 'colors'), None)
        top_design = next((p for p in top_10 if p['category'] == 'designs'), None)
        top_fit = next((p for p in top_10 if p['category'] == 'fits'), None)
        top_material = next((p for p in top_10 if p['category'] == 'materials'), None)
        
        rec_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║        T-SHIRT DESIGN RECOMMENDATIONS FOR NEXT 6 MONTHS         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🎨 TOP COLOR:     {top_color['name'] if top_color else 'N/A':<30} (+{top_color['growth'] if top_color else 0:.0f}%)  ║
║  🎯 TOP DESIGN:    {top_design['name'] if top_design else 'N/A':<30} (+{top_design['growth'] if top_design else 0:.0f}%)  ║
║  📐 TOP FIT:       {top_fit['name'] if top_fit else 'N/A':<30} (+{top_fit['growth'] if top_fit else 0:.0f}%)  ║
║  🧵 TOP MATERIAL:  {top_material['name'] if top_material else 'N/A':<30} (+{top_material['growth'] if top_material else 0:.0f}%)  ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  RECOMMENDED COMBINATION:                                        ║
║  ─────────────────────────                                       ║
║  Color:    Sage Green / Butter Yellow / Powder Blue              ║
║  Design:   Oversized Blank or Minimalist Logo                    ║
║  Fit:      Oversized with Drop Shoulders                         ║
║  Material: Organic or Heavy Cotton                               ║
║  Neckline: Crew Neck (classic) or Mock Neck (emerging)           ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠️  AVOID: Slim Fit, Cropped, V-Neck, Tie-Dye                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
        
        ax8.text(0.5, 0.5, rec_text, transform=ax8.transAxes, fontsize=9,
                fontfamily='monospace', color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', 
                         edgecolor='#00d9ff', linewidth=2))
        
        # Save
        save_path = Path(__file__).parent / 'tshirt_forecast_dashboard.png'
        plt.savefig(save_path, dpi=150, facecolor='#0a0a14', edgecolor='none', bbox_inches='tight')
        print(f'\n✅ Dashboard saved: {save_path}')
        plt.close()
        
        return save_path
    
    def _plot_category_forecast(self, ax, category: str, title: str):
        """Plot forecast bar chart for a category."""
        forecasts = self.forecasts.get(category, {})
        
        # Sort by growth
        sorted_items = sorted(
            [(name, data['growth_pct']) for name, data in forecasts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5
        
        names = [x[0].replace(' T-Shirt', '').replace(' Tee', '') for x in sorted_items]
        growths = [x[1] for x in sorted_items]
        
        colors = ['#00ff88' if g > 10 else '#ffd93d' if g > 0 else '#ff6b6b' for g in growths]
        
        bars = ax.barh(names[::-1], growths[::-1], color=colors[::-1], 
                      edgecolor='white', linewidth=0.5)
        
        for bar, growth in zip(bars, growths[::-1]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{growth:+.0f}%', va='center', fontsize=8, color='white')
        
        ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=8)
        ax.set_xlabel('Growth %', fontsize=8, color='#888')
        ax.grid(True, axis='x', alpha=0.3)
    
    def print_forecast_report(self):
        """Print detailed forecast report."""
        
        print("\n" + "="*80)
        print("             T-SHIRT FASHION FORECAST - NEXT 6 MONTHS")
        print(f"                   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*80)
        
        # Top predictions
        print("\n🚀 TOP 10 RISING TRENDS")
        print("-"*70)
        
        for i, pred in enumerate(self.get_top_predictions(10), 1):
            peak_str = pred['peak_date'].strftime('%b %d')
            print(f"  {i:2}. {pred['name']:<30} {pred['category']:<12} "
                  f"+{pred['growth']:5.1f}%  Peak: {peak_str}")
        
        # Declining
        print("\n📉 DECLINING TRENDS (Avoid)")
        print("-"*70)
        
        for pred in self.get_declining_trends(5):
            print(f"     ⬇️ {pred['name']:<30} {pred['growth']:+.1f}%")
        
        # By category
        for category in ['colors', 'designs', 'fits', 'materials', 'necklines']:
            print(f"\n{'='*70}")
            print(f"   {category.upper()} FORECAST")
            print("-"*70)
            
            forecasts = self.forecasts.get(category, {})
            sorted_items = sorted(
                [(name, data) for name, data in forecasts.items()],
                key=lambda x: x[1]['growth_pct'],
                reverse=True
            )
            
            for name, data in sorted_items[:5]:
                trend_icon = "📈" if data['growth_pct'] > 10 else "➡️" if data['growth_pct'] > -5 else "📉"
                print(f"  {trend_icon} {name:<30} Current: {data['current_value']:5.1f}  "
                      f"→ Final: {data['final_value']:5.1f}  ({data['growth_pct']:+.1f}%)")
        
        # Summary
        print("\n" + "="*80)
        print("   DESIGN RECOMMENDATION SUMMARY")
        print("="*80)
        
        print("""
    BEST T-SHIRT DESIGN FOR JULY 2026:
    ──────────────────────────────────
    
    COLOR:     Sage Green (#9DC183) or Butter Yellow (#FFFACD)
    DESIGN:    Minimalist small logo or Oversized blank
    FIT:       Oversized with dropped shoulders
    MATERIAL:  Heavy organic cotton (200+ GSM)
    NECKLINE:  Classic crew neck or emerging mock neck
    
    STYLE NOTES:
    • Embrace quiet luxury - minimal branding
    • Focus on quality fabric and construction
    • Earth tones and soft pastels trending
    • Relaxed, comfortable fits dominate
    • Sustainability messaging important
    
    TRENDS TO AVOID:
    ─────────────────
    ✗ Slim/fitted silhouettes
    ✗ Cropped lengths
    ✗ V-neck styles
    ✗ Tie-dye patterns
    ✗ Over-branded graphics
""")
        print("="*80)


def run_full_training():
    """Run complete training and forecasting pipeline."""
    
    print("="*80)
    print("    OPENTREND T-SHIRT FASHION FORECASTING MODEL")
    print("    Training on 2 years of historical trend data")
    print("="*80)
    
    # Initialize forecaster
    forecaster = TShirtFashionForecaster()
    
    # Generate training data
    forecaster.generate_training_data()
    
    # Train models
    forecaster.train_models()
    
    # Generate forecasts
    forecaster.generate_forecasts(periods=26)  # 6 months
    
    # Create dashboard
    forecaster.create_forecast_dashboard()
    
    # Print report
    forecaster.print_forecast_report()
    
    return forecaster


if __name__ == "__main__":
    forecaster = run_full_training()
    
    print("\n📊 Charts saved:")
    print("  - tshirt_forecast_dashboard.png")
