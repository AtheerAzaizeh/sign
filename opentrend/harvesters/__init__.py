"""
opentrend/harvesters/__init__.py
Global Fashion Brain - Harvester Plugin Registry
"""

from .base import (
    FashionSource,
    FashionItem,
    HarvestDatabase,
    GlobalHarvester
)

from .street_signal import RedditHarvester, PinterestHarvester
from .runway import VogueRunwayHarvester, TheImpressionHarvester
from .visual_vibe import UnsplashHarvester, PexelsHarvester
from .marketplace import DepopHarvester, GrailedHarvester

__all__ = [
    # Base classes
    'FashionSource',
    'FashionItem',
    'HarvestDatabase',
    'GlobalHarvester',
    
    # Street Signal (Social Media)
    'RedditHarvester',
    'PinterestHarvester',
    
    # Runway (High Fashion)
    'VogueRunwayHarvester',
    'TheImpressionHarvester',
    
    # Visual Vibe (Open Source)
    'UnsplashHarvester',
    'PexelsHarvester',
    
    # Marketplace (Hype)
    'DepopHarvester',
    'GrailedHarvester',
]
