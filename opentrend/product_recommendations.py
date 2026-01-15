"""
OpenTrend - Product Design Recommendations 2026
Combines color, material, and design forecasts into actionable product recommendations
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path
from datetime import datetime

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0a0a14'
plt.rcParams['axes.facecolor'] = '#12121c'

# =============================================================================
# COMBINED FORECAST DATA - Best recommendations from industry sources
# =============================================================================

PRODUCT_RECOMMENDATIONS = [
    {
        "product": "Tailored Blazer",
        "category": "Outerwear",
        "priority": 1,
        "overall_score": 96,
        "colors": [
            {"name": "Cloud Dancer", "hex": "#F0EDE5", "score": 98},
            {"name": "Transformative Teal", "hex": "#1E6B6F", "score": 95},
            {"name": "Lava Falls", "hex": "#BD3039", "score": 82},
        ],
        "materials": [
            {"name": "Heavy Sustainable Cotton", "score": 92},
            {"name": "Recycled Wool Blend", "score": 88},
            {"name": "Lyocell/Tencel", "score": 86},
        ],
        "design_elements": [
            "Quiet Luxury aesthetics",
            "Oversized/relaxed fit",
            "Minimal hardware",
            "Single-breasted, 1-2 buttons",
        ],
        "trend_alignment": "Quiet Luxury + Neo-Bourgeois"
    },
    {
        "product": "Statement Dress",
        "category": "Dresses",
        "priority": 2,
        "overall_score": 94,
        "colors": [
            {"name": "Electric Fuchsia", "hex": "#FF1493", "score": 92},
            {"name": "Sunshine Yellow", "hex": "#FFD93D", "score": 85},
            {"name": "Cloud Dancer", "hex": "#F0EDE5", "score": 98},
        ],
        "materials": [
            {"name": "Silk/Satin", "score": 90},
            {"name": "Sheer/Mesh overlay", "score": 78},
            {"name": "Bio-based synthetics", "score": 85},
        ],
        "design_elements": [
            "Bubble hem silhouette",
            "Strategic cutouts (1-2)",
            "Victorian-inspired collar",
            "Asymmetric details",
        ],
        "trend_alignment": "Victorian Revival + Bubble Hem"
    },
    {
        "product": "Reimagined Trench Coat",
        "category": "Outerwear",
        "priority": 3,
        "overall_score": 92,
        "colors": [
            {"name": "Amber Haze", "hex": "#C9A227", "score": 88},
            {"name": "Cocoa Powder", "hex": "#6B4423", "score": 85},
            {"name": "Cloud Dancer", "hex": "#F0EDE5", "score": 98},
        ],
        "materials": [
            {"name": "Technical Cotton Blend", "score": 90},
            {"name": "Water-resistant Bio-fabric", "score": 88},
            {"name": "Recycled Polyester Shell", "score": 82},
        ],
        "design_elements": [
            "Oversized proportions",
            "Dropped shoulders",
            "Belt alternatives (ties, toggles)",
            "Removable liner",
        ],
        "trend_alignment": "Sportif + Quiet Luxury"
    },
    {
        "product": "Wide-Leg Trousers",
        "category": "Bottoms",
        "priority": 4,
        "overall_score": 90,
        "colors": [
            {"name": "Cloud Dancer", "hex": "#F0EDE5", "score": 98},
            {"name": "Transformative Teal", "hex": "#1E6B6F", "score": 95},
            {"name": "Alexandrite Teal", "hex": "#1E6B6F", "score": 88},
        ],
        "materials": [
            {"name": "Heavy Linen Blend", "score": 88},
            {"name": "Organic Cotton Twill", "score": 90},
            {"name": "Tencel Drape", "score": 86},
        ],
        "design_elements": [
            "High-waisted (transition low-rise)",
            "Full-length, floor-grazing",
            "Pleated front",
            "Side pockets, minimal hardware",
        ],
        "trend_alignment": "Quiet Luxury + Oversized Volume"
    },
    {
        "product": "Sculptural Skirt",
        "category": "Bottoms",
        "priority": 5,
        "overall_score": 88,
        "colors": [
            {"name": "Electric Fuchsia", "hex": "#FF1493", "score": 92},
            {"name": "Blue Aura", "hex": "#7B9DC4", "score": 85},
            {"name": "Dusty Rose", "hex": "#D4A5A5", "score": 78},
        ],
        "materials": [
            {"name": "Structured Taffeta", "score": 85},
            {"name": "Recycled Nylon", "score": 82},
            {"name": "Organic Cotton", "score": 88},
        ],
        "design_elements": [
            "Bubble hem",
            "A-line with volume",
            "Asymmetric hemline",
            "High waist with wide band",
        ],
        "trend_alignment": "Bubble Hem + Statement Pieces"
    },
    {
        "product": "Technical Hoodie",
        "category": "Activewear",
        "priority": 6,
        "overall_score": 86,
        "colors": [
            {"name": "Transformative Teal", "hex": "#1E6B6F", "score": 95},
            {"name": "Jelly Mint", "hex": "#98FF98", "score": 78},
            {"name": "Cloud Dancer", "hex": "#F0EDE5", "score": 98},
        ],
        "materials": [
            {"name": "Recycled Performance Fleece", "score": 90},
            {"name": "Bio-based Elastane Blend", "score": 88},
            {"name": "Organic French Terry", "score": 85},
        ],
        "design_elements": [
            "Oversized fit",
            "Dropped shoulders", 
            "Zippered hidden pockets (Guardian Design)",
            "Minimal branding",
        ],
        "trend_alignment": "Sportif + Guardian Design"
    },
    {
        "product": "Victorian Blouse",
        "category": "Tops",
        "priority": 7,
        "overall_score": 85,
        "colors": [
            {"name": "Cloud Dancer", "hex": "#F0EDE5", "score": 98},
            {"name": "Tea Rose", "hex": "#F1C5C5", "score": 75},
            {"name": "Fresh Purple", "hex": "#8A2BE2", "score": 82},
        ],
        "materials": [
            {"name": "Organic Silk", "score": 92},
            {"name": "Recycled Polyester Chiffon", "score": 80},
            {"name": "Tencel Satin", "score": 88},
        ],
        "design_elements": [
            "High ruffled collar",
            "Puff sleeves (subtle)",
            "Pearl button details",
            "Layering-friendly",
        ],
        "trend_alignment": "Victorian Revival + Neo-Bourgeois"
    },
    {
        "product": "Gold Statement Jewelry Set",
        "category": "Accessories",
        "priority": 8,
        "overall_score": 88,
        "colors": [
            {"name": "Brushed Gold", "hex": "#D4AF37", "score": 90},
            {"name": "Rose Gold", "hex": "#B76E79", "score": 82},
        ],
        "materials": [
            {"name": "Recycled Gold Plating", "score": 92},
            {"name": "Brass with Gold Finish", "score": 85},
        ],
        "design_elements": [
            "Chunky chains",
            "Layered necklaces",
            "Stacked bangles",
            "Sculptural earrings",
        ],
        "trend_alignment": "Going for Gold + Maximalist Accents"
    },
]

# Combined color palette recommendation
TOP_COLOR_PALETTE = [
    {"name": "Cloud Dancer", "hex": "#F0EDE5", "usage": "Base/Neutral", "score": 98},
    {"name": "Transformative Teal", "hex": "#1E6B6F", "usage": "Statement/Accent", "score": 95},
    {"name": "Electric Fuchsia", "hex": "#FF1493", "usage": "Bold Accent", "score": 92},
    {"name": "Amber Haze", "hex": "#C9A227", "usage": "Warm Accent", "score": 88},
    {"name": "Lava Falls", "hex": "#BD3039", "usage": "Power Color", "score": 82},
    {"name": "Blue Aura", "hex": "#7B9DC4", "usage": "Soft Accent", "score": 85},
]

# Top material recommendations
TOP_MATERIALS = [
    {"name": "Bio-based Fabrics", "score": 92, "applications": "All categories"},
    {"name": "Recycled Cotton (GRS)", "score": 90, "applications": "Casual, Bottoms"},
    {"name": "Lyocell/Tencel", "score": 88, "applications": "Dresses, Blouses"},
    {"name": "Organic Silk", "score": 86, "applications": "Luxury pieces"},
    {"name": "Technical Performance", "score": 85, "applications": "Activewear, Outerwear"},
    {"name": "Recycled Wool", "score": 84, "applications": "Outerwear, Suiting"},
]


def create_product_recommendations_dashboard():
    """Create comprehensive product recommendation dashboard."""
    
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('OPENTREND 2026 PRODUCT DESIGN RECOMMENDATIONS\nOptimal Color, Material & Design Combinations', 
                 fontsize=22, fontweight='bold', color='white', y=0.98)
    
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3,
                          left=0.04, right=0.96, top=0.93, bottom=0.04)
    
    # ==========================================================================
    # TOP ROW: Color Palette & Materials
    # ==========================================================================
    
    # Color Palette
    ax_colors = fig.add_subplot(gs[0, :2])
    ax_colors.set_title('RECOMMENDED COLOR PALETTE 2026', fontsize=14, fontweight='bold', color='white', pad=15)
    
    for i, c in enumerate(TOP_COLOR_PALETTE):
        rect = Rectangle((i*0.16, 0.2), 0.14, 0.6, facecolor=c['hex'], edgecolor='white', linewidth=2)
        ax_colors.add_patch(rect)
        text_color = 'white' if c['hex'] in ['#1E6B6F', '#BD3039', '#FF1493'] else '#333'
        ax_colors.text(i*0.16 + 0.07, 0.5, c['name'].replace(' ', '\n'), ha='center', va='center', 
                      fontsize=8, color=text_color, fontweight='bold')
        ax_colors.text(i*0.16 + 0.07, 0.1, c['usage'], ha='center', va='center', 
                      fontsize=7, color='white')
    
    ax_colors.set_xlim(0, 1)
    ax_colors.set_ylim(0, 1)
    ax_colors.axis('off')
    
    # Materials
    ax_materials = fig.add_subplot(gs[0, 2:])
    ax_materials.set_title('TOP RECOMMENDED MATERIALS', fontsize=14, fontweight='bold', color='white', pad=15)
    
    mat_names = [m['name'] for m in TOP_MATERIALS]
    mat_scores = [m['score'] for m in TOP_MATERIALS]
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(TOP_MATERIALS)))[::-1]
    
    bars = ax_materials.barh(mat_names[::-1], mat_scores[::-1], color=colors, edgecolor='white', linewidth=0.5)
    ax_materials.set_xlabel('Score', fontsize=10, color='#888')
    ax_materials.set_xlim(75, 100)
    ax_materials.grid(True, axis='x', alpha=0.3)
    
    for bar, mat in zip(bars, TOP_MATERIALS[::-1]):
        ax_materials.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                         mat['applications'], va='center', fontsize=8, color='#aaa')
    
    # ==========================================================================
    # PRODUCT CARDS (2x4 grid)
    # ==========================================================================
    
    product_positions = [
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
    ]
    
    for idx, (row, col) in enumerate(product_positions):
        if idx >= len(PRODUCT_RECOMMENDATIONS):
            break
            
        prod = PRODUCT_RECOMMENDATIONS[idx]
        ax = fig.add_subplot(gs[row, col])
        
        # Product card background
        ax.set_facecolor('#1a1a2e')
        
        # Title
        ax.set_title(f"#{prod['priority']} {prod['product']}", fontsize=11, fontweight='bold', 
                    color='white', pad=8)
        
        # Score badge
        score_color = '#00ff88' if prod['overall_score'] >= 90 else '#ffd93d' if prod['overall_score'] >= 80 else '#ff6b6b'
        ax.text(0.95, 0.95, f"{prod['overall_score']}", transform=ax.transAxes, fontsize=16, 
               fontweight='bold', color=score_color, ha='right', va='top',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='#2d3a4f', edgecolor=score_color))
        
        # Colors strip
        y_pos = 0.85
        ax.text(0.05, y_pos, "COLORS:", transform=ax.transAxes, fontsize=8, color='#888', fontweight='bold')
        for j, c in enumerate(prod['colors'][:3]):
            rect = Rectangle((0.25 + j*0.22, y_pos - 0.08), 0.18, 0.12, 
                            transform=ax.transAxes, facecolor=c['hex'], edgecolor='white', linewidth=1)
            ax.add_patch(rect)
        
        # Materials
        y_pos = 0.65
        ax.text(0.05, y_pos, "MATERIALS:", transform=ax.transAxes, fontsize=8, color='#888', fontweight='bold')
        for j, m in enumerate(prod['materials'][:2]):
            ax.text(0.05, y_pos - 0.12 - j*0.1, f"• {m['name']}", transform=ax.transAxes, 
                   fontsize=7, color='white')
        
        # Design elements
        y_pos = 0.35
        ax.text(0.05, y_pos, "DESIGN:", transform=ax.transAxes, fontsize=8, color='#888', fontweight='bold')
        for j, d in enumerate(prod['design_elements'][:3]):
            ax.text(0.05, y_pos - 0.1 - j*0.08, f"• {d}", transform=ax.transAxes, 
                   fontsize=6, color='#ccc')
        
        # Trend alignment
        ax.text(0.05, 0.02, f"Trend: {prod['trend_alignment']}", transform=ax.transAxes, 
               fontsize=6, color='#00d9ff', style='italic')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # ==========================================================================
    # BOTTOM ROW: Summary & Action Items
    # ==========================================================================
    
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    summary_text = """
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                           2026 FASHION PRODUCT STRATEGY SUMMARY                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  PRIMARY COLORS                         PRIMARY MATERIALS                        KEY DESIGN DIRECTIONS                     PRIORITY PRODUCTS               │
│  ─────────────────                      ───────────────────                       ─────────────────────                     ─────────────────               │
│  1. Cloud Dancer (Base)                 1. Bio-based Fabrics                      1. Quiet Luxury aesthetic                1. Tailored Blazer (96)         │
│  2. Transformative Teal (Statement)     2. Recycled Cotton (GRS)                  2. Oversized/relaxed silhouettes         2. Statement Dress (94)         │
│  3. Electric Fuchsia (Bold)             3. Lyocell/Tencel                         3. Victorian-inspired details            3. Reimagined Trench (92)       │
│  4. Amber Haze (Warm)                   4. Organic Silk                           4. Bubble hem silhouettes                4. Wide-Leg Trousers (90)       │
│  5. Lava Falls (Power)                  5. Technical Performance                  5. Strategic cutouts                     5. Sculptural Skirt (88)        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  AVOID: Over-branding, Skinny silhouettes, Fast-fashion synthetics, Y2K revival elements, Excessive puff sleeves                                          │
│  FOCUS: Sustainability credentials, Quality over quantity, Versatile seasonless pieces, Minimal elegant hardware                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes, fontsize=9,
                   fontfamily='monospace', color='white', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#00d9ff', linewidth=2))
    
    # Save
    save_path = Path(__file__).parent / 'product_recommendations_2026.png'
    plt.savefig(save_path, dpi=150, facecolor='#0a0a14', edgecolor='none', bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()
    
    return save_path


def create_single_product_card(product_name: str = "Tailored Blazer"):
    """Create a detailed single product recommendation card."""
    
    # Find the product
    prod = next((p for p in PRODUCT_RECOMMENDATIONS if p['product'] == product_name), PRODUCT_RECOMMENDATIONS[0])
    
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.suptitle(f"PRODUCT DESIGN RECOMMENDATION", fontsize=16, fontweight='bold', color='white', y=0.98)
    
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    
    # Product name
    ax.text(0.5, 0.92, prod['product'].upper(), transform=ax.transAxes, fontsize=24, 
           fontweight='bold', color='white', ha='center')
    
    ax.text(0.5, 0.87, f"Category: {prod['category']} | Priority: #{prod['priority']} | Score: {prod['overall_score']}/100",
           transform=ax.transAxes, fontsize=12, color='#888', ha='center')
    
    # Color swatches
    ax.text(0.1, 0.78, "RECOMMENDED COLORS", transform=ax.transAxes, fontsize=12, 
           fontweight='bold', color='#00d9ff')
    
    for i, c in enumerate(prod['colors']):
        rect = Rectangle((0.1 + i*0.28, 0.62), 0.24, 0.12, transform=ax.transAxes,
                         facecolor=c['hex'], edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        text_color = 'white' if c['hex'] in ['#1E6B6F', '#BD3039', '#FF1493', '#6B4423'] else '#333'
        ax.text(0.1 + i*0.28 + 0.12, 0.68, c['name'], transform=ax.transAxes,
               fontsize=9, color=text_color, ha='center', fontweight='bold')
        ax.text(0.1 + i*0.28 + 0.12, 0.60, f"Score: {c['score']}", transform=ax.transAxes,
               fontsize=8, color='#888', ha='center')
    
    # Materials
    ax.text(0.1, 0.52, "RECOMMENDED MATERIALS", transform=ax.transAxes, fontsize=12,
           fontweight='bold', color='#00d9ff')
    
    for i, m in enumerate(prod['materials']):
        ax.text(0.1, 0.46 - i*0.05, f"• {m['name']} (Score: {m['score']})", 
               transform=ax.transAxes, fontsize=11, color='white')
    
    # Design elements
    ax.text(0.1, 0.30, "DESIGN ELEMENTS", transform=ax.transAxes, fontsize=12,
           fontweight='bold', color='#00d9ff')
    
    for i, d in enumerate(prod['design_elements']):
        ax.text(0.1, 0.24 - i*0.04, f"✓ {d}", transform=ax.transAxes, fontsize=10, color='#ccc')
    
    # Trend alignment
    ax.text(0.1, 0.08, f"TREND ALIGNMENT: {prod['trend_alignment']}", transform=ax.transAxes,
           fontsize=11, color='#ffd93d', style='italic')
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent / f'product_card_{product_name.lower().replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150, facecolor='#0a0a14', edgecolor='none', bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()
    
    return save_path


def print_product_recommendations():
    """Print detailed product recommendations to console."""
    
    print("\n" + "="*90)
    print("                    OPENTREND 2026 PRODUCT DESIGN RECOMMENDATIONS")
    print("                  Optimal Color, Material & Design Combinations")
    print("="*90)
    
    for prod in PRODUCT_RECOMMENDATIONS:
        print(f"\n{'─'*90}")
        print(f"#{prod['priority']} {prod['product'].upper()} | Score: {prod['overall_score']}/100 | {prod['category']}")
        print(f"{'─'*90}")
        
        print("\n  COLORS:")
        for c in prod['colors']:
            print(f"    • {c['name']:<25} {c['hex']}  (Score: {c['score']})")
        
        print("\n  MATERIALS:")
        for m in prod['materials']:
            print(f"    • {m['name']:<35} (Score: {m['score']})")
        
        print("\n  DESIGN ELEMENTS:")
        for d in prod['design_elements']:
            print(f"    ✓ {d}")
        
        print(f"\n  TREND ALIGNMENT: {prod['trend_alignment']}")
    
    print("\n" + "="*90)
    print("SUMMARY: TOP COMBINATIONS")
    print("="*90)
    print("""
    BEST COLOR + MATERIAL COMBINATIONS:
    
    1. Cloud Dancer + Organic Silk = Perfect for Victorian Blouses
    2. Transformative Teal + Bio-based Fabric = Ideal for Statement Dresses
    3. Electric Fuchsia + Recycled Satin = Bold Evening Wear
    4. Amber Haze + Technical Cotton = Reimagined Trench Coats
    5. Lava Falls + Recycled Wool = Power Suiting
    
    BEST PRODUCT + DESIGN COMBINATIONS:
    
    1. Tailored Blazer + Quiet Luxury + Cloud Dancer = Business to Evening
    2. Statement Dress + Bubble Hem + Fuchsia = Event/Party Wear
    3. Trench Coat + Sportif + Amber Haze = Everyday Luxury
    4. Wide-Leg Trousers + Teal + Organic Cotton = Versatile Wardrobe Staple
    5. Technical Hoodie + Guardian Design + Teal = Elevated Streetwear
""")
    print("="*90)


if __name__ == "__main__":
    print("="*80)
    print("OPENTREND - Product Design Recommendations")
    print("="*80)
    
    print("\nGenerating product recommendations dashboard...")
    create_product_recommendations_dashboard()
    
    print("\nGenerating individual product cards...")
    create_single_product_card("Tailored Blazer")
    create_single_product_card("Statement Dress")
    create_single_product_card("Technical Hoodie")
    
    print("\nGenerating report...")
    print_product_recommendations()
    
    print("\n" + "="*80)
    print("Charts saved:")
    print("  - product_recommendations_2026.png")
    print("  - product_card_tailored_blazer.png")
    print("  - product_card_statement_dress.png")
    print("  - product_card_technical_hoodie.png")
    print("="*80)
