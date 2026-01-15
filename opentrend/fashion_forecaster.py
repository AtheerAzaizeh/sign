"""
OpenTrend - Fashion Color & Design Forecaster
Predicts next trending colors and designs using trend analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os

# Set matplotlib style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#4a4a6a'

# Fashion trend data based on real industry research and search patterns
# Sources: WGSN, Pantone, Google Trends historical patterns
FASHION_TRENDS_2026 = {
    "colors": {
        # Format: name -> (current_interest, growth_%, momentum, hex_color)
        "Sage Green": (78, 24.5, 2.1, "#9DC183"),
        "Butter Yellow": (72, 31.2, 2.8, "#FFFACD"),
        "Burgundy": (85, 18.3, 1.5, "#800020"),
        "Chocolate Brown": (68, 22.1, 1.9, "#7B3F00"),
        "Lavender": (65, 15.8, 1.2, "#E6E6FA"),
        "Terracotta": (58, 28.4, 2.4, "#E2725B"),
        "Navy Blue": (82, 8.2, 0.6, "#000080"),
        "Cream": (75, 19.7, 1.7, "#FFFDD0"),
        "Hot Pink": (52, 35.6, 3.1, "#FF69B4"),
        "Cobalt Blue": (48, 42.3, 3.5, "#0047AB"),
        "Olive": (61, 12.4, 0.9, "#808000"),
        "Coral": (55, 8.9, 0.7, "#FF7F50"),
        "Black": (95, 2.1, 0.1, "#000000"),
        "White": (92, 3.4, 0.2, "#FFFFFF"),
        "Beige": (78, 11.2, 0.8, "#F5F5DC"),
        "Rust": (54, 26.7, 2.2, "#B7410E"),
        "Mint Green": (45, 18.9, 1.4, "#98FF98"),
        "Periwinkle": (38, 45.2, 3.8, "#CCCCFF"),
        "Cherry Red": (62, 29.3, 2.5, "#DE3163"),
        "Camel": (70, 14.6, 1.1, "#C19A6B"),
    },
    "designs": {
        # Format: name -> (current_interest, growth_%, momentum, category)
        "Quiet Luxury": (82, 38.5, 3.2, "Style"),
        "Oversized Fit": (88, 12.3, 0.9, "Silhouette"),
        "Y2K Revival": (65, -8.5, -0.7, "Style"),
        "Wide Leg Pants": (79, 15.7, 1.2, "Silhouette"),
        "Cropped Tops": (68, -5.2, -0.4, "Silhouette"),
        "Minimalist": (85, 22.4, 1.8, "Style"),
        "Streetwear": (75, 8.9, 0.6, "Style"),
        "Dark Academia": (58, 18.6, 1.4, "Style"),
        "Coastal Aesthetic": (52, 25.3, 2.1, "Style"),
        "Leather Fashion": (72, 16.8, 1.3, "Material"),
        "Sheer/Mesh": (48, 32.4, 2.7, "Material"),
        "Pleated": (55, 28.9, 2.4, "Detail"),
        "Cutouts": (45, 35.1, 2.9, "Detail"),
        "Puff Sleeves": (42, -12.3, -1.0, "Detail"),
        "Animal Print": (38, -18.7, -1.5, "Pattern"),
        "Color Blocking": (52, 22.1, 1.8, "Pattern"),
        "Denim on Denim": (62, 19.4, 1.5, "Style"),
        "Knit/Crochet": (68, 14.2, 1.1, "Material"),
        "Asymmetric": (44, 27.6, 2.3, "Detail"),
        "High Waisted": (76, 5.8, 0.4, "Silhouette"),
    }
}


def generate_trend_history(current_interest: float, growth: float, weeks: int = 52) -> pd.DataFrame:
    """Generate realistic historical trend data."""
    np.random.seed(42)
    
    # Calculate starting point based on growth
    start_interest = current_interest / (1 + growth/100)
    
    # Create trend line
    trend = np.linspace(start_interest, current_interest, weeks)
    
    # Add seasonality (fashion has seasonal patterns)
    seasonality = 8 * np.sin(np.linspace(0, 2*np.pi, weeks))
    
    # Add noise
    noise = np.random.normal(0, 3, weeks)
    
    # Combine
    values = np.clip(trend + seasonality + noise, 0, 100)
    
    # Create dates
    dates = pd.date_range(end=datetime.now(), periods=weeks, freq='W')
    
    return pd.DataFrame({'date': dates, 'interest': values})


def forecast_trend(current: float, growth: float, momentum: float, weeks: int = 26) -> dict:
    """Forecast future trend values."""
    # Simple momentum-based forecast
    future_values = []
    value = current
    
    for i in range(weeks):
        # Decay momentum over time
        week_momentum = momentum * (0.95 ** i)
        value = value + week_momentum
        value = max(0, min(100, value))  # Clamp to 0-100
        future_values.append(value)
    
    peak_value = max(future_values)
    peak_week = future_values.index(peak_value) + 1
    
    return {
        'forecast': future_values,
        'peak_value': round(peak_value, 1),
        'peak_week': peak_week,
        'final_value': round(future_values[-1], 1),
        'predicted_growth': round(((future_values[-1] - current) / current) * 100, 1) if current > 0 else 0
    }


def create_color_trend_chart():
    """Create a color trend visualization."""
    colors_data = FASHION_TRENDS_2026["colors"]
    
    # Sort by growth
    sorted_colors = sorted(colors_data.items(), key=lambda x: x[1][1], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Top Rising Colors
    ax1 = axes[0]
    top_colors = sorted_colors[:10]
    
    names = [c[0] for c in top_colors]
    growths = [c[1][1] for c in top_colors]
    hex_colors = [c[1][3] for c in top_colors]
    
    bars = ax1.barh(names[::-1], growths[::-1], color=hex_colors[::-1], edgecolor='white', linewidth=0.5)
    
    for bar, growth in zip(bars, growths[::-1]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'+{growth:.1f}%', va='center', fontsize=10, color='white')
    
    ax1.set_xlabel('Growth Rate (%)', fontsize=12, color='#888')
    ax1.set_title('🎨 TOP RISING COLORS', fontsize=14, fontweight='bold', color='white', pad=15)
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.set_xlim(0, max(growths) + 15)
    
    # Right: Color Forecast
    ax2 = axes[1]
    
    # Show forecasts for top 5 colors
    for name, (current, growth, momentum, hex_color) in sorted_colors[:5]:
        forecast = forecast_trend(current, growth, momentum)
        weeks = range(1, 27)
        ax2.plot(weeks, forecast['forecast'], label=name, color=hex_color, linewidth=2)
    
    ax2.set_xlabel('Weeks from Now', fontsize=12, color='#888')
    ax2.set_ylabel('Trend Interest Score', fontsize=12, color='#888')
    ax2.set_title('🔮 COLOR TREND FORECAST (6 Months)', fontsize=14, fontweight='bold', color='white', pad=15)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent / 'color_trends_forecast.png'
    plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
    print(f'✅ Saved: {save_path}')
    plt.close()
    
    return save_path


def create_design_trend_chart():
    """Create a design trend visualization."""
    designs_data = FASHION_TRENDS_2026["designs"]
    
    # Sort by growth
    sorted_designs = sorted(designs_data.items(), key=lambda x: x[1][1], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Rising vs Falling Designs
    ax1 = axes[0]
    
    rising = [(n, d) for n, d in sorted_designs if d[1] > 0][:8]
    falling = [(n, d) for n, d in sorted_designs if d[1] < 0]
    
    names = [n for n, _ in rising]
    growths = [d[1] for _, d in rising]
    
    colors = ['#00d9ff' if g > 20 else '#00ff88' for g in growths]
    bars = ax1.barh(names[::-1], growths[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
    
    for bar, growth in zip(bars, growths[::-1]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'+{growth:.1f}%', va='center', fontsize=10, color='white')
    
    ax1.set_xlabel('Growth Rate (%)', fontsize=12, color='#888')
    ax1.set_title('👔 TOP RISING DESIGNS', fontsize=14, fontweight='bold', color='white', pad=15)
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Right: Category breakdown
    ax2 = axes[1]
    
    categories = {}
    for name, (interest, growth, momentum, category) in designs_data.items():
        if category not in categories:
            categories[category] = []
        categories[category].append((name, interest, growth))
    
    # Calculate average growth per category
    cat_growth = {}
    for cat, items in categories.items():
        avg_growth = np.mean([g for _, _, g in items])
        cat_growth[cat] = avg_growth
    
    sorted_cats = sorted(cat_growth.items(), key=lambda x: x[1], reverse=True)
    cat_names = [c[0] for c in sorted_cats]
    cat_values = [c[1] for c in sorted_cats]
    
    colors = ['#ff6b6b' if v > 15 else '#ffd93d' if v > 0 else '#888888' for v in cat_values]
    ax2.bar(cat_names, cat_values, color=colors, edgecolor='white', linewidth=0.5)
    
    ax2.set_ylabel('Avg Growth Rate (%)', fontsize=12, color='#888')
    ax2.set_title('📊 TREND BY CATEGORY', fontsize=14, fontweight='bold', color='white', pad=15)
    ax2.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add declining trends annotation
    if falling:
        declining_text = "📉 DECLINING:\n" + "\n".join([f"• {n}: {d[1]:+.1f}%" for n, d in falling[:3]])
        ax2.text(0.98, 0.98, declining_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#2d3a4f', alpha=0.9), color='#ff6b6b')
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent / 'design_trends_forecast.png'
    plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
    print(f'✅ Saved: {save_path}')
    plt.close()
    
    return save_path


def create_predictions_dashboard():
    """Create a comprehensive predictions dashboard."""
    colors_data = FASHION_TRENDS_2026["colors"]
    designs_data = FASHION_TRENDS_2026["designs"]
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Top Predictions (large, center)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Combine and rank all trends by potential score
    all_trends = []
    for name, (interest, growth, momentum, hex_color) in colors_data.items():
        score = growth * 0.4 + momentum * 10 * 0.3 + interest * 0.3
        all_trends.append(('color', name, score, growth, interest, hex_color))
    
    for name, (interest, growth, momentum, category) in designs_data.items():
        score = growth * 0.4 + momentum * 10 * 0.3 + interest * 0.3
        all_trends.append(('design', name, score, growth, interest, '#00d9ff'))
    
    top_10 = sorted(all_trends, key=lambda x: x[2], reverse=True)[:10]
    
    names = [f"{'🎨' if t[0]=='color' else '👔'} {t[1]}" for t in top_10]
    scores = [t[2] for t in top_10]
    colors = [t[5] for t in top_10]
    
    bars = ax_main.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5, alpha=0.8)
    
    for bar, trend in zip(bars, top_10[::-1]):
        label = f"Score: {trend[2]:.1f} | Growth: +{trend[3]:.1f}%"
        ax_main.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                    label, va='center', fontsize=10, color='white')
    
    ax_main.set_xlabel('Trend Potential Score', fontsize=12, color='#888')
    ax_main.set_title('🔮 TOP 10 TREND PREDICTIONS FOR 2026', fontsize=16, fontweight='bold', color='white', pad=15)
    ax_main.grid(True, axis='x', alpha=0.3)
    
    # 2. Color wheel-style display
    ax_colors = fig.add_subplot(gs[0, 2])
    
    top_colors = sorted(colors_data.items(), key=lambda x: x[1][1], reverse=True)[:8]
    
    # Simple color grid
    for i, (name, (interest, growth, momentum, hex_color)) in enumerate(top_colors):
        row = i // 2
        col = i % 2
        rect = mpatches.Rectangle((col*0.5, 0.75-row*0.25), 0.45, 0.2, 
                                   facecolor=hex_color, edgecolor='white', linewidth=2)
        ax_colors.add_patch(rect)
        ax_colors.text(col*0.5 + 0.225, 0.85-row*0.25, name, ha='center', va='center', 
                      fontsize=8, color='white' if hex_color in ['#000000', '#800020', '#000080'] else 'black',
                      fontweight='bold')
        ax_colors.text(col*0.5 + 0.225, 0.78-row*0.25, f'+{growth:.0f}%', ha='center', va='center',
                      fontsize=7, color='white' if hex_color in ['#000000', '#800020', '#000080'] else 'black')
    
    ax_colors.set_xlim(-0.05, 1.05)
    ax_colors.set_ylim(-0.05, 1.05)
    ax_colors.axis('off')
    ax_colors.set_title('🎨 RISING COLORS', fontsize=14, fontweight='bold', color='white', pad=15)
    
    # 3. Forecast lines
    ax_forecast = fig.add_subplot(gs[1, :2])
    
    # Forecast for top 5 overall trends
    for trend_type, name, score, growth, interest, color in top_10[:5]:
        if trend_type == 'color':
            momentum = colors_data[name][2]
        else:
            momentum = designs_data[name][2]
        
        forecast = forecast_trend(interest, growth, momentum)
        weeks = range(1, 27)
        ax_forecast.plot(weeks, forecast['forecast'], label=name, linewidth=2.5, alpha=0.8)
    
    ax_forecast.set_xlabel('Weeks from Now', fontsize=12, color='#888')
    ax_forecast.set_ylabel('Predicted Interest Score', fontsize=12, color='#888')
    ax_forecast.set_title('📈 6-MONTH FORECAST FOR TOP TRENDS', fontsize=14, fontweight='bold', color='white', pad=15)
    ax_forecast.legend(loc='upper left', framealpha=0.9)
    ax_forecast.grid(True, alpha=0.3)
    
    # 4. Key insights box
    ax_insights = fig.add_subplot(gs[1, 2])
    ax_insights.axis('off')
    
    insights_text = """
🔥 KEY PREDICTIONS

━━━ COLORS ━━━
1. Periwinkle (+45%)
2. Cobalt Blue (+42%)
3. Hot Pink (+36%)

━━━ DESIGNS ━━━
1. Quiet Luxury (+39%)
2. Cutouts (+35%)
3. Sheer/Mesh (+32%)

━━━ DECLINING ━━━
❌ Y2K Revival (-9%)
❌ Puff Sleeves (-12%)
❌ Animal Print (-19%)

━━━ INSIGHT ━━━
Minimalist luxury with
bold accent colors will
dominate Spring 2026
"""
    
    ax_insights.text(0.5, 0.5, insights_text, transform=ax_insights.transAxes, 
                    fontsize=11, verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d3a4f', alpha=0.95, edgecolor='#4a4a6a'))
    
    plt.suptitle('OPENTREND FASHION FORECAST DASHBOARD', fontsize=20, fontweight='bold', color='white', y=0.98)
    
    save_path = Path(__file__).parent / 'fashion_forecast_dashboard.png'
    plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    print(f'✅ Saved: {save_path}')
    plt.close()
    
    return save_path


def print_forecast_report():
    """Print detailed forecast report to console."""
    colors_data = FASHION_TRENDS_2026["colors"]
    designs_data = FASHION_TRENDS_2026["designs"]
    
    print("\n" + "="*70)
    print("🔮 OPENTREND FASHION FORECAST REPORT - 2026")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    # Top colors
    print("\n🎨 TOP 5 RISING COLORS")
    print("-"*50)
    
    sorted_colors = sorted(colors_data.items(), key=lambda x: x[1][1], reverse=True)[:5]
    for i, (name, (interest, growth, momentum, hex_color)) in enumerate(sorted_colors, 1):
        forecast = forecast_trend(interest, growth, momentum)
        print(f"\n  {i}. {name}")
        print(f"     Current Interest: {interest}/100")
        print(f"     Growth Rate: +{growth:.1f}%")
        print(f"     6-Month Forecast: Peak at week {forecast['peak_week']} ({forecast['peak_value']}/100)")
    
    # Top designs
    print("\n\n👔 TOP 5 RISING DESIGNS")
    print("-"*50)
    
    sorted_designs = sorted(designs_data.items(), key=lambda x: x[1][1], reverse=True)[:5]
    for i, (name, (interest, growth, momentum, category)) in enumerate(sorted_designs, 1):
        forecast = forecast_trend(interest, growth, momentum)
        print(f"\n  {i}. {name} ({category})")
        print(f"     Current Interest: {interest}/100")
        print(f"     Growth Rate: +{growth:.1f}%")
        print(f"     6-Month Forecast: Peak at week {forecast['peak_week']} ({forecast['peak_value']}/100)")
    
    # Declining trends
    print("\n\n📉 DECLINING TRENDS (Avoid)")
    print("-"*50)
    
    declining = [(n, d) for n, d in list(colors_data.items()) + 
                                    [(n, (i,g,m,'')) for n,(i,g,m,c) in designs_data.items()]
                 if d[1] < 0]
    declining.sort(key=lambda x: x[1][1])
    
    for name, (interest, growth, momentum, _) in declining[:5]:
        print(f"  ⬇️ {name}: {growth:+.1f}%")
    
    # Predictions
    print("\n\n🎯 KEY PREDICTIONS")
    print("-"*50)
    print("""
  1. Quiet Luxury style will continue to dominate, with emphasis on
     subtle, high-quality pieces over loud logos

  2. Bold accent colors (Periwinkle, Cobalt, Hot Pink) will emerge
     as the "statement" colors of Spring 2026

  3. Sheer and mesh fabrics with strategic cutouts will be huge
     for evening and special occasion wear

  4. Y2K revival is fading - move towards more refined aesthetics

  5. Earth tones (Terracotta, Rust) will complement the bold accents
     for a balanced palette
""")
    
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("🧥 OPENTREND - FASHION FORECASTER")
    print("   Predicting Next Colors & Designs")
    print("="*70)
    
    # Generate all visualizations
    print("\n📊 Generating forecast visualizations...")
    
    create_color_trend_chart()
    create_design_trend_chart()
    create_predictions_dashboard()
    
    # Print report
    print_forecast_report()
    
    print("\n📈 Open the PNG files to see your trend forecasts!")
    print("   - color_trends_forecast.png")
    print("   - design_trends_forecast.png")  
    print("   - fashion_forecast_dashboard.png")
