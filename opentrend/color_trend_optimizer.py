"""
opentrend/color_trend_optimizer.py
Targeted Color Analysis for Product-Level Demand Forecasting

This module determines the "Winning Color" for each winning product
by analyzing Google Trends data for specific color+product combinations.

Strategy:
- Use WGSN/Pantone 2026 color forecasts (no generic black/white)
- Test only top products × key seasonal colors
- Calculate relative growth (slope) not just volume
- Minimize API calls: 5 products × 6 colors = 30 queries max

Usage:
    optimizer = ColorOptimizer()
    
    # Add winning products
    optimizer.set_products(["linen shirt", "crochet dress", "midi skirt"])
    
    # Run analysis
    results = optimizer.analyze()
    
    # Generate merchandising matrix
    optimizer.print_matrix()
    optimizer.save_results("product_color_matrix.json")
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Import OpenTrend modules
try:
    from signals.trend_tracker import TrendTracker
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from signals.trend_tracker import TrendTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 2026 COLOR PALETTE (WGSN/Pantone Inspired)
# =============================================================================

# Key trending colors for 2026 - avoid generic black/white/grey
# These are fashion-forward colors with trend signal value
PALETTE_2026 = [
    {
        "name": "Butter Yellow",
        "hex": "#FFFACD",
        "category": "warm",
        "keywords": ["butter yellow", "pale yellow", "soft yellow"]
    },
    {
        "name": "Sage Green", 
        "hex": "#9DC183",
        "category": "neutral",
        "keywords": ["sage green", "sage", "muted green"]
    },
    {
        "name": "Digital Lavender",
        "hex": "#E6E6FA",
        "category": "cool",
        "keywords": ["digital lavender", "lavender", "light purple"]
    },
    {
        "name": "Terracotta",
        "hex": "#E2725B",
        "category": "warm",
        "keywords": ["terracotta", "burnt orange", "rust"]
    },
    {
        "name": "Cobalt Blue",
        "hex": "#0047AB",
        "category": "cool",
        "keywords": ["cobalt blue", "electric blue", "bright blue"]
    },
    {
        "name": "Espresso Brown",
        "hex": "#4E3629",
        "category": "neutral",
        "keywords": ["espresso brown", "chocolate brown", "dark brown"]
    },
    {
        "name": "Cherry Red",
        "hex": "#DE3163",
        "category": "warm",
        "keywords": ["cherry red", "bright red", "lipstick red"]
    },
]

# Simple color name list for quick access
COLORS_2026 = [c["name"] for c in PALETTE_2026]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ColorResult:
    """Result for a single color+product combination."""
    color: str
    product: str
    query: str
    growth_percent: float = 0.0
    trend_direction: str = "unknown"
    data_points: int = 0
    last_interest: float = 0.0
    status: str = "pending"  # pending, success, failed
    error: Optional[str] = None
    
    @property
    def rank_score(self) -> float:
        """Score for ranking (higher = better)."""
        if self.status != "success":
            return -999
        return self.growth_percent


@dataclass
class ProductColorAnalysis:
    """Complete color analysis for a single product."""
    product: str
    colors: List[ColorResult] = field(default_factory=list)
    analyzed_at: str = ""
    
    def __post_init__(self):
        if not self.analyzed_at:
            self.analyzed_at = datetime.now().isoformat()
    
    @property
    def top_color(self) -> Optional[ColorResult]:
        """Get the top-performing color."""
        successful = [c for c in self.colors if c.status == "success"]
        if not successful:
            return None
        return max(successful, key=lambda c: c.growth_percent)
    
    @property
    def ranked_colors(self) -> List[ColorResult]:
        """Get colors ranked by growth (descending)."""
        successful = [c for c in self.colors if c.status == "success"]
        return sorted(successful, key=lambda c: c.growth_percent, reverse=True)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "product": self.product,
            "analyzed_at": self.analyzed_at,
            "top_color": self.top_color.color if self.top_color else None,
            "top_growth": self.top_color.growth_percent if self.top_color else None,
            "colors": [asdict(c) for c in self.ranked_colors]
        }


# =============================================================================
# COLOR OPTIMIZER
# =============================================================================

class ColorOptimizer:
    """
    Targeted color analysis for product-level demand forecasting.
    
    Determines the "Winning Color" for each winning product by
    analyzing Google Trends data for specific color+product combinations.
    
    Strategy:
    - Only test WGSN/Pantone 2026 colors (skip generic black/white)
    - Limit to top 5 products × 6-7 colors = ~35 requests max
    - Use relative growth (slope) not just volume
    - Rank colors per product for merchandising decisions
    """
    
    # Rate limiting
    DELAY_SECONDS = 60  # Google allows ~1 request per minute
    RATE_LIMIT_BACKOFF = 300  # 5 min on 429
    
    def __init__(
        self,
        products: List[str] = None,
        colors: List[str] = None,
        delay_seconds: int = 60,
        timeframe: str = 'today 12-m'
    ):
        """
        Initialize the optimizer.
        
        Args:
            products: List of winning products to analyze
            colors: List of colors to test (default: 2026 palette)
            delay_seconds: Delay between API requests
            timeframe: Google Trends timeframe
        """
        self.products = products or []
        self.colors = colors or COLORS_2026
        self.delay_seconds = max(delay_seconds, 30)
        self.timeframe = timeframe
        
        self.tracker = TrendTracker(initial_delay=5.0, retries=2)
        self.results: Dict[str, ProductColorAnalysis] = {}
        
        logger.info(f"ColorOptimizer initialized: {len(self.colors)} colors, {delay_seconds}s delay")
    
    def set_products(self, products: List[str]):
        """Set the products to analyze."""
        self.products = [p.lower().strip() for p in products]
        logger.info(f"Set {len(self.products)} products: {self.products}")
    
    def load_products_from_report(self, report_path: str = None, top_n: int = 5):
        """
        Load top winning products from batch miner report.
        
        Args:
            report_path: Path to spring_2026_forecast.md
            top_n: Number of top products to load
        """
        if report_path is None:
            report_path = Path(__file__).parent / "spring_2026_forecast.md"
        
        # This would parse the markdown - for now use defaults
        # TODO: Implement markdown parsing
        
        logger.warning("Report parsing not implemented. Use set_products() instead.")
    
    def _generate_query(self, color: str, product: str) -> str:
        """Generate search query for color+product combination."""
        return f"{color.lower()} {product.lower()}"
    
    def _countdown(self, seconds: int, message: str = "Next request"):
        """Show countdown timer."""
        for remaining in tqdm(range(seconds, 0, -1),
                              desc=f"⏳ {message}",
                              ncols=60,
                              leave=False):
            time.sleep(1)
    
    def analyze(self, max_products: int = 5) -> Dict[str, ProductColorAnalysis]:
        """
        Run color analysis for all products.
        
        Args:
            max_products: Maximum number of products to analyze
            
        Returns:
            Dict mapping product -> ProductColorAnalysis
        """
        products = self.products[:max_products]
        
        if not products:
            logger.warning("No products set. Use set_products() first.")
            return {}
        
        total_queries = len(products) * len(self.colors)
        estimated_time = total_queries * self.delay_seconds // 60
        
        logger.info("=" * 60)
        logger.info("🎨 STARTING COLOR TREND ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"   Products: {len(products)}")
        logger.info(f"   Colors: {len(self.colors)}")
        logger.info(f"   Total queries: {total_queries}")
        logger.info(f"   Estimated time: ~{estimated_time} minutes")
        logger.info("=" * 60)
        
        for product in products:
            logger.info(f"\n📦 Analyzing: {product.upper()}")
            logger.info("-" * 40)
            
            analysis = ProductColorAnalysis(product=product)
            
            for i, color in enumerate(self.colors):
                query = self._generate_query(color, product)
                logger.info(f"   [{i+1}/{len(self.colors)}] {query}")
                
                result = ColorResult(
                    color=color,
                    product=product,
                    query=query
                )
                
                try:
                    # Fetch trend data
                    trend_result = self.tracker.analyze_keyword(
                        query,
                        timeframe=self.timeframe
                    )
                    
                    if trend_result['status'] == 'success':
                        metrics = trend_result['metrics']
                        result.growth_percent = metrics['growth_percent']
                        result.trend_direction = metrics['trend']
                        result.data_points = len(trend_result['data'])
                        result.last_interest = trend_result['data'][-1]['interest'] if trend_result['data'] else 0
                        result.status = "success"
                        
                        # Log result
                        emoji = "📈" if result.growth_percent > 0 else "📉"
                        logger.info(f"       {emoji} {result.growth_percent:+.1f}%")
                    else:
                        result.status = "failed"
                        result.error = trend_result.get('error', 'No data')
                        logger.warning(f"       ⚠️ {result.error}")
                        
                except Exception as e:
                    result.status = "failed"
                    result.error = str(e)[:50]
                    
                    if '429' in str(e):
                        logger.warning(f"       🚫 Rate limited. Waiting {self.RATE_LIMIT_BACKOFF}s...")
                        self._countdown(self.RATE_LIMIT_BACKOFF, "Rate limit cooldown")
                    else:
                        logger.error(f"       ❌ {result.error}")
                
                analysis.colors.append(result)
                
                # Wait before next request
                if i < len(self.colors) - 1:
                    self._countdown(self.delay_seconds, "Next color")
            
            self.results[product] = analysis
            
            # Summary for this product
            if analysis.top_color:
                logger.info(f"   🏆 Winner: {analysis.top_color.color} ({analysis.top_color.growth_percent:+.1f}%)")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ COLOR ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
        return self.results
    
    def print_matrix(self):
        """Print the merchandising matrix."""
        print("\n" + "=" * 70)
        print("🎨 PRODUCT-COLOR MERCHANDISING MATRIX")
        print("   Spring/Summer 2026 Recommendations")
        print("=" * 70)
        
        for product, analysis in self.results.items():
            print(f"\n📦 PRODUCT: {product.upper()}")
            print("-" * 50)
            
            ranked = analysis.ranked_colors
            
            if not ranked:
                print("   No successful color data")
                continue
            
            for i, color_result in enumerate(ranked, 1):
                growth = color_result.growth_percent
                
                # Determine emoji and label
                if i == 1:
                    emoji = "🏆"
                    label = "PRODUCTION PRIORITY"
                elif growth > 10:
                    emoji = "📈"
                    label = "Strong Growth"
                elif growth > 0:
                    emoji = "➡️"
                    label = "Stable"
                else:
                    emoji = "📉"
                    label = "Declining"
                
                print(f"   {i}. {emoji} {color_result.color:<20} ({growth:+.1f}% Growth) - {label}")
        
        print("\n" + "=" * 70)
    
    def save_results(self, output_path: str = None) -> str:
        """
        Save results to JSON file.
        
        Args:
            output_path: Path for output file
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = Path(__file__).parent / "product_color_matrix.json"
        else:
            output_path = Path(output_path)
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "palette": COLORS_2026,
            "products": {
                product: analysis.to_dict()
                for product, analysis in self.results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_path}")
        return str(output_path)
    
    def generate_markdown_report(self, output_path: str = None) -> str:
        """
        Generate a Markdown merchandising report.
        
        Args:
            output_path: Path for the report
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = Path(__file__).parent / "color_merchandising_report.md"
        
        report = f"""# 🎨 Color Merchandising Matrix - Spring/Summer 2026

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Data Source:** Google Trends (12-month analysis)  
**Color Palette:** WGSN/Pantone 2026 Forecast

---

## Quick Summary

| Product | Top Color | Growth | Recommendation |
|---------|-----------|--------|----------------|
"""
        
        for product, analysis in self.results.items():
            top = analysis.top_color
            if top:
                rec = "🟢 Prioritize" if top.growth_percent > 10 else "🟡 Consider"
                report += f"| {product.title()} | **{top.color}** | {top.growth_percent:+.1f}% | {rec} |\n"
        
        report += "\n---\n\n## Detailed Analysis\n"
        
        for product, analysis in self.results.items():
            report += f"\n### {product.title()}\n\n"
            report += "| Rank | Color | Growth | Interest | Signal |\n"
            report += "|------|-------|--------|----------|--------|\n"
            
            for i, cr in enumerate(analysis.ranked_colors, 1):
                signal = "🔥" if cr.growth_percent > 20 else "📈" if cr.growth_percent > 0 else "📉"
                report += f"| {i} | {cr.color} | {cr.growth_percent:+.1f}% | {cr.last_interest:.0f}/100 | {signal} |\n"
        
        report += f"""

---

## 2026 Color Palette Reference

| Color | Hex | Category |
|-------|-----|----------|
"""
        for c in PALETTE_2026:
            report += f"| {c['name']} | `{c['hex']}` | {c['category'].title()} |\n"
        
        report += "\n*Report generated by OpenTrend Color Optimizer*\n"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {output_path}")
        return str(output_path)


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          OPENTREND COLOR OPTIMIZER                                    ║
║          Spring/Summer 2026 Color Analysis                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Analyzes color trends for winning products to determine              ║
║  optimal color allocation for merchandising.                          ║
║                                                                        ║
║  2026 Color Palette:                                                   ║
║  🟡 Butter Yellow  🟢 Sage Green  💜 Digital Lavender                 ║
║  🟠 Terracotta     🔵 Cobalt Blue  🟤 Espresso Brown  🔴 Cherry Red   ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize optimizer
    optimizer = ColorOptimizer(delay_seconds=60)
    
    # Set winning products (from batch miner results)
    winning_products = [
        "linen shirt",
        "crochet dress",
        "midi skirt",
        "wide leg pants",
        "halter top"
    ]
    
    optimizer.set_products(winning_products)
    
    # Show estimated time
    total = len(winning_products) * len(COLORS_2026)
    est_mins = total * 60 // 60
    
    print(f"\n📊 Analysis Plan:")
    print(f"   Products: {len(winning_products)}")
    print(f"   Colors: {len(COLORS_2026)}")
    print(f"   Total queries: {total}")
    print(f"   Estimated time: ~{est_mins} minutes")
    
    print("\n" + "=" * 60)
    response = input("Start color analysis? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        # Run analysis
        optimizer.analyze()
        
        # Print results
        optimizer.print_matrix()
        
        # Save results
        optimizer.save_results()
        optimizer.generate_markdown_report()
    else:
        print("Cancelled.")
