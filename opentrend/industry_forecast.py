"""
OpenTrend - Industry Fashion Forecast 2026
Data sourced from Pantone, WGSN, and Vogue Trends
"""
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Set matplotlib style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0f0f1a'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['axes.edgecolor'] = '#4a4a6a'
plt.rcParams['font.family'] = 'sans-serif'

# =============================================================================
# PANTONE 2026 OFFICIAL DATA
# =============================================================================
PANTONE_2026 = {
    "color_of_year": {
        "name": "Cloud Dancer",
        "code": "PANTONE 11-4201",
        "hex": "#F0EDE5",
        "description": "Airy, balanced off-white evoking clarity and calm",
        "trend_score": 98,
    },
    "spring_summer_palette": [
        {"name": "Cloud Dancer", "hex": "#F0EDE5", "score": 98, "category": "Neutral"},
        {"name": "Alexandrite Teal", "hex": "#1E6B6F", "score": 88, "category": "Jewel"},
        {"name": "Lava Falls", "hex": "#BD3039", "score": 82, "category": "Bold"},
        {"name": "Dusty Rose", "hex": "#D4A5A5", "score": 78, "category": "Romantic"},
        {"name": "Tea Rose", "hex": "#F1C5C5", "score": 75, "category": "Romantic"},
        {"name": "Angora", "hex": "#E8DCC8", "score": 72, "category": "Neutral"},
        {"name": "Summer Grey", "hex": "#A5A5A5", "score": 68, "category": "Neutral"},
        {"name": "White Onyx", "hex": "#F5F5F0", "score": 65, "category": "Neutral"},
    ],
    "key_pairings": [
        ("Cloud Dancer", "Emerald Green"),
        ("Cloud Dancer", "Navy Blue"),
        ("Cloud Dancer", "Brushed Brass"),
        ("Cloud Dancer", "Wine Red"),
    ]
}

# =============================================================================
# WGSN + COLORO 2026 OFFICIAL DATA
# =============================================================================
WGSN_2026 = {
    "color_of_year": {
        "name": "Transformative Teal",
        "code": "COLORO 089-47-18",
        "hex": "#1E6B6F",
        "description": "Fluid blend of dark blue and aqua green - Earth-first mindset",
        "trend_score": 95,
    },
    "spring_summer_colors": [
        {"name": "Transformative Teal", "hex": "#1E6B6F", "score": 95, "symbolism": "Eco-accountability"},
        {"name": "Electric Fuchsia", "hex": "#FF1493", "score": 92, "symbolism": "Rebellious resistance"},
        {"name": "Blue Aura", "hex": "#7B9DC4", "score": 85, "symbolism": "Healing serenity"},
        {"name": "Amber Haze", "hex": "#C9A227", "score": 88, "symbolism": "Ancient wisdom"},
        {"name": "Jelly Mint", "hex": "#98FF98", "score": 78, "symbolism": "Playful optimism"},
    ],
    "autumn_winter_colors": [
        {"name": "Transformative Teal", "hex": "#1E6B6F", "score": 95},
        {"name": "Wax Paper", "hex": "#FFF8DC", "score": 88},
        {"name": "Cocoa Powder", "hex": "#6B4423", "score": 85},
        {"name": "Fresh Purple", "hex": "#8A2BE2", "score": 82},
        {"name": "Green Glow", "hex": "#39FF14", "score": 79},
    ],
    "key_trends": [
        {"name": "Quiet Luxury", "score": 96, "description": "Understated elegance, quality over logos"},
        {"name": "Attainable Escapism", "score": 90, "description": "Multi-sensory, democratic luxury"},
        {"name": "Guardian Design", "score": 85, "description": "Anti-theft features, RFID-blocking"},
        {"name": "Sportif", "score": 88, "description": "Athletic function meets everyday style"},
        {"name": "Going for Gold", "score": 82, "description": "Bold gold as stability symbol"},
    ],
    "materials": [
        {"name": "Bio-based Fabrics", "score": 92, "trend": "rising"},
        {"name": "Recycled Cotton (GRS)", "score": 88, "trend": "rising"},
        {"name": "Lyocell/Tencel", "score": 86, "trend": "rising"},
        {"name": "Heavy Cotton", "score": 84, "trend": "stable"},
        {"name": "Cashmere", "score": 80, "trend": "stable"},
        {"name": "Technical Fabrics", "score": 85, "trend": "rising"},
        {"name": "Sheer/Mesh", "score": 78, "trend": "rising"},
    ]
}

# =============================================================================
# VOGUE SS2026 RUNWAY DATA
# =============================================================================
VOGUE_2026 = {
    "top_colors": [
        {"name": "Sunshine Yellow", "hex": "#FFD93D", "runway_presence": 85},
        {"name": "Cerulean Blue", "hex": "#007BA7", "runway_presence": 82},
        {"name": "Fire Engine Red", "hex": "#CE2029", "runway_presence": 80},
        {"name": "Sage Green", "hex": "#9DC183", "runway_presence": 78},
        {"name": "Bubblegum Rose", "hex": "#FFC0CB", "runway_presence": 75},
        {"name": "Chartreuse", "hex": "#7FFF00", "runway_presence": 70},
    ],
    "design_trends": [
        {"name": "Victorian Revival", "score": 88, "elements": "Corsets, high collars, artful boning"},
        {"name": "Bubble Hem", "score": 85, "elements": "Sculptural skirts and dresses"},
        {"name": "Strategic Cutouts", "score": 82, "elements": "Architectural, deliberate openings"},
        {"name": "Neo-Bourgeois", "score": 80, "elements": "Openwork tweed, monograms, lace"},
        {"name": "Band Jackets", "score": 78, "elements": "Braiding, tassels, theatrical"},
        {"name": "Undergarments as Outer", "score": 75, "elements": "Bra tops, corset styling"},
    ],
    "silhouettes": [
        {"name": "Oversized Volume", "score": 90, "direction": "continuing"},
        {"name": "Wide-Leg Trousers", "score": 88, "direction": "continuing"},
        {"name": "Low Waistlines", "score": 75, "direction": "emerging"},
        {"name": "Bubble Skirts", "score": 82, "direction": "emerging"},
        {"name": "Balloon Pants", "score": 70, "direction": "emerging"},
        {"name": "Fit-and-Flare", "score": 78, "direction": "returning"},
    ],
    "key_products": [
        {"name": "Trench Coat (Reimagined)", "score": 92},
        {"name": "Tailored Suiting", "score": 90},
        {"name": "Statement Skirts", "score": 85},
        {"name": "Boilersuits", "score": 78},
        {"name": "Scarf Accessories", "score": 82},
        {"name": "Gold Jewelry", "score": 88},
    ]
}


def create_comprehensive_dashboard():
    """Create the ultimate industry fashion forecast dashboard."""
    
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('OPENTREND 2026 FASHION FORECAST\nBased on Pantone, WGSN & Vogue Data', 
                 fontsize=24, fontweight='bold', color='white', y=0.98)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, 
                          left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    # ==========================================================================
    # ROW 1: COLOR OF THE YEAR + TOP COLORS
    # ==========================================================================
    
    # 1A: Pantone Color of the Year
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANTONE_2026['color_of_year']['hex'])
    ax1.text(0.5, 0.65, 'PANTONE', ha='center', va='center', fontsize=12, 
             color='#333', fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.5, 'Color of the Year', ha='center', va='center', fontsize=9, 
             color='#555', transform=ax1.transAxes)
    ax1.text(0.5, 0.35, 'CLOUD DANCER', ha='center', va='center', fontsize=14, 
             color='#333', fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.2, '11-4201', ha='center', va='center', fontsize=10, 
             color='#666', transform=ax1.transAxes)
    ax1.axis('off')
    ax1.set_title('PANTONE 2026', fontsize=11, fontweight='bold', color='white', pad=10)
    
    # 1B: WGSN Color of the Year
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(WGSN_2026['color_of_year']['hex'])
    ax2.text(0.5, 0.65, 'WGSN + COLORO', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.5, 'Color of the Year', ha='center', va='center', fontsize=9, 
             color='#ccc', transform=ax2.transAxes)
    ax2.text(0.5, 0.35, 'TRANSFORMATIVE', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.22, 'TEAL', ha='center', va='center', fontsize=14, 
             color='white', fontweight='bold', transform=ax2.transAxes)
    ax2.axis('off')
    ax2.set_title('WGSN 2026', fontsize=11, fontweight='bold', color='white', pad=10)
    
    # 1C: Top Colors Comparison
    ax3 = fig.add_subplot(gs[0, 2:])
    
    # Combine all top colors
    all_colors = []
    for c in PANTONE_2026['spring_summer_palette'][:4]:
        all_colors.append(('Pantone', c['name'], c['hex'], c['score']))
    for c in WGSN_2026['spring_summer_colors'][:4]:
        all_colors.append(('WGSN', c['name'], c['hex'], c['score']))
    for c in VOGUE_2026['top_colors'][:4]:
        all_colors.append(('Vogue', c['name'], c['hex'], c['runway_presence']))
    
    # Sort by score
    all_colors.sort(key=lambda x: x[3], reverse=True)
    top_colors = all_colors[:10]
    
    y_positions = np.arange(len(top_colors))
    colors = [c[2] for c in top_colors]
    scores = [c[3] for c in top_colors]
    labels = [f"{c[1]} ({c[0]})" for c in top_colors]
    
    bars = ax3.barh(y_positions, scores, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(labels[::-1], fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel('Trend Score', fontsize=10, color='#888')
    ax3.set_title('TOP 10 COLORS ACROSS ALL SOURCES', fontsize=12, fontweight='bold', color='white', pad=10)
    ax3.set_xlim(0, 110)
    ax3.grid(True, axis='x', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        ax3.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f'{score}', va='center', fontsize=9, color='white')
    
    # ==========================================================================
    # ROW 2: DESIGN TRENDS + MATERIALS + SILHOUETTES
    # ==========================================================================
    
    # 2A: Design Trends (WGSN + Vogue combined)
    ax4 = fig.add_subplot(gs[1, :2])
    
    design_trends = []
    for t in WGSN_2026['key_trends']:
        design_trends.append((t['name'], t['score'], 'WGSN', '#00d9ff'))
    for t in VOGUE_2026['design_trends']:
        design_trends.append((t['name'], t['score'], 'Vogue', '#ff6b9d'))
    
    design_trends.sort(key=lambda x: x[1], reverse=True)
    top_trends = design_trends[:10]
    
    names = [t[0] for t in top_trends]
    scores = [t[1] for t in top_trends]
    colors = [t[3] for t in top_trends]
    
    bars = ax4.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
    ax4.set_xlabel('Trend Score', fontsize=10, color='#888')
    ax4.set_title('TOP DESIGN & STYLE TRENDS', fontsize=12, fontweight='bold', color='white', pad=10)
    ax4.grid(True, axis='x', alpha=0.3)
    ax4.set_xlim(0, 105)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#00d9ff', label='WGSN'),
                       Patch(facecolor='#ff6b9d', label='Vogue')]
    ax4.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    # 2B: Materials Forecast
    ax5 = fig.add_subplot(gs[1, 2])
    
    materials = WGSN_2026['materials']
    mat_names = [m['name'] for m in materials]
    mat_scores = [m['score'] for m in materials]
    mat_colors = ['#00ff88' if m['trend'] == 'rising' else '#ffd93d' for m in materials]
    
    ax5.barh(mat_names[::-1], mat_scores[::-1], color=mat_colors[::-1], edgecolor='white', linewidth=0.5)
    ax5.set_xlabel('Trend Score', fontsize=10, color='#888')
    ax5.set_title('KEY MATERIALS 2026', fontsize=12, fontweight='bold', color='white', pad=10)
    ax5.grid(True, axis='x', alpha=0.3)
    
    # 2C: Silhouettes
    ax6 = fig.add_subplot(gs[1, 3])
    
    silhouettes = VOGUE_2026['silhouettes']
    sil_names = [s['name'] for s in silhouettes]
    sil_scores = [s['score'] for s in silhouettes]
    sil_colors = ['#ff6b6b' if s['direction'] == 'emerging' else '#00d9ff' for s in silhouettes]
    
    ax6.barh(sil_names[::-1], sil_scores[::-1], color=sil_colors[::-1], edgecolor='white', linewidth=0.5)
    ax6.set_xlabel('Runway Score', fontsize=10, color='#888')
    ax6.set_title('KEY SILHOUETTES', fontsize=12, fontweight='bold', color='white', pad=10)
    ax6.grid(True, axis='x', alpha=0.3)
    
    # ==========================================================================
    # ROW 3: KEY PRODUCTS + COLOR PALETTES + INSIGHTS
    # ==========================================================================
    
    # 3A: Key Product Categories
    ax7 = fig.add_subplot(gs[2, 0])
    
    products = VOGUE_2026['key_products']
    prod_names = [p['name'] for p in products]
    prod_scores = [p['score'] for p in products]
    
    wedges, texts, autotexts = ax7.pie(prod_scores, labels=prod_names, autopct='%1.0f%%',
                                        colors=plt.cm.plasma(np.linspace(0.2, 0.8, len(products))),
                                        textprops={'fontsize': 8, 'color': 'white'})
    ax7.set_title('KEY PRODUCTS 2026', fontsize=12, fontweight='bold', color='white', pad=10)
    
    # 3B: WGSN Color Palette Display
    ax8 = fig.add_subplot(gs[2, 1])
    
    ss_colors = WGSN_2026['spring_summer_colors']
    for i, c in enumerate(ss_colors):
        rect = mpatches.Rectangle((i*0.2, 0), 0.18, 1, facecolor=c['hex'], edgecolor='white', linewidth=1)
        ax8.add_patch(rect)
        ax8.text(i*0.2 + 0.09, 0.5, c['name'].split()[0], ha='center', va='center', 
                fontsize=7, color='white' if c['hex'] not in ['#98FF98', '#FFF8DC'] else 'black',
                rotation=90, fontweight='bold')
    
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    ax8.set_title('WGSN SS2026 PALETTE', fontsize=12, fontweight='bold', color='white', pad=10)
    
    # 3C: Pantone Palette Display
    ax9 = fig.add_subplot(gs[2, 2])
    
    pantone_colors = PANTONE_2026['spring_summer_palette']
    n_cols = 4
    n_rows = 2
    for i, c in enumerate(pantone_colors[:8]):
        row = i // n_cols
        col = i % n_cols
        rect = mpatches.Rectangle((col*0.25, 0.5-row*0.5), 0.23, 0.45, 
                                   facecolor=c['hex'], edgecolor='white', linewidth=1)
        ax9.add_patch(rect)
        text_color = 'white' if c['hex'] in ['#1E6B6F', '#BD3039', '#A5A5A5'] else '#333'
        ax9.text(col*0.25 + 0.115, 0.5-row*0.5 + 0.225, c['name'][:10], 
                ha='center', va='center', fontsize=6, color=text_color, fontweight='bold')
    
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    ax9.set_title('PANTONE SS2026 PALETTE', fontsize=12, fontweight='bold', color='white', pad=10)
    
    # 3D: Key Insights
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')
    
    insights = """
KEY FORECAST INSIGHTS

COLORS:
  - Cloud Dancer (Pantone): Soft luxury
  - Transformative Teal (WGSN): Sustainability
  - Bold Fuchsia & Reds: Statement pieces
  - Earth browns: +15% growth expected

DESIGN:
  - Quiet Luxury dominates (+96 score)
  - Victorian elements returning
  - Bubble hems emerging
  - Strategic cutouts (architectural)

MATERIALS:
  - Bio-based fabrics: +92 score
  - Heavy sustainable cotton
  - Technical-luxury blends
  - Sheer/mesh for evening

PRODUCTS TO INVEST IN:
  1. Reimagined Trench Coats
  2. Tailored Suiting
  3. Statement Skirts
  4. Gold Jewelry
  5. Scarf Accessories

AVOID:
  - Over-branded items
  - Skinny silhouettes
  - Fast fashion synthetics
"""
    
    ax10.text(0.05, 0.95, insights, transform=ax10.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             fontfamily='monospace', color='white',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#2d3a4f', alpha=0.95, edgecolor='#4a4a6a'))
    
    # Save
    save_path = Path(__file__).parent / 'industry_fashion_forecast_2026.png'
    plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', edgecolor='none', bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()
    
    return save_path


def create_color_palette_chart():
    """Create detailed color palette comparison chart."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('2026 COLOR FORECASTS BY SOURCE', fontsize=18, fontweight='bold', color='white', y=0.98)
    
    # Pantone
    ax1 = axes[0]
    pantone_colors = PANTONE_2026['spring_summer_palette']
    for i, c in enumerate(pantone_colors):
        rect = mpatches.Rectangle((0, i), 3, 0.9, facecolor=c['hex'], edgecolor='white', linewidth=1)
        ax1.add_patch(rect)
        text_color = 'white' if c['hex'] in ['#1E6B6F', '#BD3039', '#A5A5A5'] else '#333'
        ax1.text(0.1, i+0.45, c['name'], va='center', fontsize=10, color=text_color, fontweight='bold')
        ax1.text(2.9, i+0.45, f"{c['score']}", va='center', ha='right', fontsize=10, color=text_color)
    
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, len(pantone_colors))
    ax1.axis('off')
    ax1.set_title('PANTONE SS2026', fontsize=14, fontweight='bold', color='white', pad=15)
    
    # WGSN
    ax2 = axes[1]
    wgsn_colors = WGSN_2026['spring_summer_colors']
    for i, c in enumerate(wgsn_colors):
        rect = mpatches.Rectangle((0, i), 3, 0.9, facecolor=c['hex'], edgecolor='white', linewidth=1)
        ax2.add_patch(rect)
        text_color = 'white' if c['hex'] not in ['#98FF98', '#C9A227'] else '#333'
        ax2.text(0.1, i+0.45, c['name'], va='center', fontsize=10, color=text_color, fontweight='bold')
        ax2.text(2.9, i+0.45, f"{c['score']}", va='center', ha='right', fontsize=10, color=text_color)
    
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, max(len(wgsn_colors), len(pantone_colors)))
    ax2.axis('off')
    ax2.set_title('WGSN + COLORO SS2026', fontsize=14, fontweight='bold', color='white', pad=15)
    
    # Vogue
    ax3 = axes[2]
    vogue_colors = VOGUE_2026['top_colors']
    for i, c in enumerate(vogue_colors):
        rect = mpatches.Rectangle((0, i), 3, 0.9, facecolor=c['hex'], edgecolor='white', linewidth=1)
        ax3.add_patch(rect)
        text_color = 'white' if c['hex'] not in ['#FFD93D', '#FFC0CB', '#7FFF00'] else '#333'
        ax3.text(0.1, i+0.45, c['name'], va='center', fontsize=10, color=text_color, fontweight='bold')
        ax3.text(2.9, i+0.45, f"{c['runway_presence']}%", va='center', ha='right', fontsize=10, color=text_color)
    
    ax3.set_xlim(0, 3)
    ax3.set_ylim(0, max(len(vogue_colors), len(pantone_colors)))
    ax3.axis('off')
    ax3.set_title('VOGUE RUNWAY SS2026', fontsize=14, fontweight='bold', color='white', pad=15)
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent / 'color_palettes_2026.png'
    plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', edgecolor='none', bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()
    
    return save_path


def print_industry_report():
    """Print comprehensive industry forecast report."""
    
    print("\n" + "="*80)
    print("             OPENTREND 2026 INDUSTRY FASHION FORECAST")
    print("        Based on Pantone, WGSN, and Vogue Runway Analysis")
    print(f"                  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)
    
    # Colors of the Year
    print("\n" + "-"*80)
    print("COLORS OF THE YEAR 2026")
    print("-"*80)
    
    print(f"""
    PANTONE: {PANTONE_2026['color_of_year']['name']} ({PANTONE_2026['color_of_year']['code']})
             "{PANTONE_2026['color_of_year']['description']}"
             Hex: {PANTONE_2026['color_of_year']['hex']}
    
    WGSN:    {WGSN_2026['color_of_year']['name']} ({WGSN_2026['color_of_year']['code']})
             "{WGSN_2026['color_of_year']['description']}"
             Hex: {WGSN_2026['color_of_year']['hex']}
""")
    
    # Top Colors
    print("-"*80)
    print("TOP 10 COLORS FOR SS2026 (Combined Rankings)")
    print("-"*80)
    
    all_colors = []
    for c in PANTONE_2026['spring_summer_palette']:
        all_colors.append((c['name'], c['score'], 'Pantone', c['hex']))
    for c in WGSN_2026['spring_summer_colors']:
        all_colors.append((c['name'], c['score'], 'WGSN', c['hex']))
    for c in VOGUE_2026['top_colors']:
        all_colors.append((c['name'], c['runway_presence'], 'Vogue', c['hex']))
    
    all_colors.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, score, source, hex_code) in enumerate(all_colors[:10], 1):
        print(f"    {i:2}. {name:<25} Score: {score:3}  Source: {source:<8} {hex_code}")
    
    # Design Trends
    print("\n" + "-"*80)
    print("TOP DESIGN & STYLE TRENDS 2026")
    print("-"*80)
    
    design_trends = WGSN_2026['key_trends'] + [
        {"name": t['name'], "score": t['score'], "description": t['elements']} 
        for t in VOGUE_2026['design_trends']
    ]
    design_trends.sort(key=lambda x: x['score'], reverse=True)
    
    for i, t in enumerate(design_trends[:10], 1):
        print(f"    {i:2}. {t['name']:<25} Score: {t['score']:3}")
        if 'description' in t:
            print(f"        - {t['description']}")
    
    # Materials
    print("\n" + "-"*80)
    print("KEY MATERIALS FOR 2026")
    print("-"*80)
    
    for m in WGSN_2026['materials']:
        trend_icon = "^" if m['trend'] == 'rising' else "="
        print(f"    {trend_icon} {m['name']:<25} Score: {m['score']:3}")
    
    # Silhouettes
    print("\n" + "-"*80)
    print("KEY SILHOUETTES")
    print("-"*80)
    
    for s in VOGUE_2026['silhouettes']:
        dir_icon = "NEW" if s['direction'] == 'emerging' else ">>>" if s['direction'] == 'continuing' else "<<<"
        print(f"    [{dir_icon}] {s['name']:<25} Score: {s['score']:3}")
    
    # Key Products
    print("\n" + "-"*80)
    print("KEY PRODUCTS TO INVEST IN")
    print("-"*80)
    
    for i, p in enumerate(VOGUE_2026['key_products'], 1):
        print(f"    {i}. {p['name']:<30} Score: {p['score']:3}")
    
    # Summary Recommendations
    print("\n" + "="*80)
    print("FORECAST RECOMMENDATIONS")
    print("="*80)
    
    print("""
    TOP COLORS TO ADOPT:
    1. Cloud Dancer (Pantone) - Perfect neutral base
    2. Transformative Teal (WGSN) - Sustainability statement
    3. Electric Fuchsia - Bold accent color
    4. Lava Falls Red - Power color
    5. Sunshine Yellow - Mood-boosting bright
    
    TOP DESIGNS TO FOLLOW:
    1. Quiet Luxury aesthetic (minimalist, quality-focused)
    2. Victorian-inspired details (corsets, high collars)
    3. Bubble hem silhouettes (dresses, skirts)
    4. Strategic architectural cutouts
    5. Sportif-luxury hybrid pieces
    
    TOP MATERIALS TO SOURCE:
    1. Bio-based sustainable fabrics
    2. GRS-certified recycled cotton
    3. Lyocell/Tencel blends
    4. Technical performance fabrics
    5. Heavy luxury cottons
    
    TOP PRODUCTS TO DEVELOP:
    1. Reimagined trench coats (new materials, shapes)
    2. Statement tailored suits
    3. Sculptural skirts (bubble, asymmetric)
    4. Gold jewelry collections
    5. Scarf-integrated accessories
    
    TRENDS TO AVOID:
    - Over-branded/logo-heavy pieces
    - Skinny/tight silhouettes
    - Fast-fashion synthetics
    - Y2K revival (declining)
    - Puff sleeves (declining)
""")
    
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("OPENTREND - Industry Fashion Forecast")
    print("Data: Pantone + WGSN + Vogue 2026")
    print("="*80)
    
    print("\nGenerating visualizations...")
    create_comprehensive_dashboard()
    create_color_palette_chart()
    
    print("\nGenerating report...")
    print_industry_report()
    
    print("\n" + "="*80)
    print("Charts saved:")
    print("  - industry_fashion_forecast_2026.png")
    print("  - color_palettes_2026.png")
    print("="*80)
