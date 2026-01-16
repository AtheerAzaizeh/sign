"""
opentrend/harvesters/runway.py
"Runway" Harvester - High Fashion Sources

Scrapes the latest collections from fashion week coverage.
Sources: Vogue Runway, The Impression, Fashion Week Online

These sources provide early signals of designer-level trends
that trickle down to mass market 6-18 months later.
"""

from __future__ import annotations

import re
import time
import random
import logging
from typing import Generator, List, Optional
from urllib.parse import urljoin
from datetime import datetime

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from .base import FashionSource, FashionItem

logger = logging.getLogger(__name__)


class VogueRunwayHarvester(FashionSource):
    """
    Harvests runway imagery from Vogue Runway.
    
    Scrapes:
    - Latest fashion week collections
    - Designer lookbooks
    - Trend reports
    
    Uses respectful scraping with rate limiting and
    User-Agent rotation to minimize detection.
    """
    
    BASE_URL = "https://www.vogue.com"
    RUNWAY_URL = "https://www.vogue.com/fashion-shows"
    
    # Seasons to target
    SEASONS = [
        'spring-2026-ready-to-wear',
        'fall-2026-ready-to-wear',
        'spring-2026-menswear',
        'resort-2026',
        'pre-fall-2026'
    ]
    
    def __init__(
        self,
        max_designers: int = 20,
        looks_per_designer: int = 30,
        data_dir: str = "./data/harvested"
    ):
        super().__init__(data_dir)
        self.max_designers = max_designers
        self.looks_per_designer = looks_per_designer
        
    @property
    def name(self) -> str:
        return "vogue_runway"
    
    def validate(self) -> bool:
        """Validate Vogue access."""
        if not BS4_AVAILABLE:
            logger.warning("beautifulsoup4 not installed. Run: pip install beautifulsoup4")
            return False
        
        try:
            response = self.session.get(self.BASE_URL, timeout=15)
            if response.status_code == 403:
                logger.warning("Vogue blocked request. Rotating User-Agent.")
                self.rotate_user_agent()
                response = self.session.get(self.BASE_URL, timeout=15)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Vogue validation failed: {e}")
            return False
    
    def _get_season_designers(self, season: str) -> List[dict]:
        """Get list of designers for a season."""
        designers = []
        
        try:
            url = f"{self.RUNWAY_URL}/{season}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return designers
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find designer links
            links = soup.find_all('a', href=re.compile(r'/fashion-shows/.*' + season))
            
            for link in links[:self.max_designers]:
                designer_name = link.get_text(strip=True)
                designer_url = urljoin(self.BASE_URL, link['href'])
                
                if designer_name and '/slideshow/' not in designer_url:
                    designers.append({
                        'name': designer_name,
                        'url': designer_url,
                        'season': season
                    })
            
        except Exception as e:
            logger.warning(f"Error getting designers for {season}: {e}")
        
        return designers
    
    def _get_collection_looks(self, designer_url: str) -> List[dict]:
        """Get look images from a designer's collection page."""
        looks = []
        
        try:
            # Try the collection slideshow
            slideshow_url = f"{designer_url}/slideshow/collection"
            response = self.session.get(slideshow_url, timeout=15)
            
            if response.status_code != 200:
                # Fall back to main page
                response = self.session.get(designer_url, timeout=15)
            
            if response.status_code != 200:
                return looks
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find runway images
            img_tags = soup.find_all('img')
            
            for img in img_tags[:self.looks_per_designer]:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy')
                
                if src and ('runway' in src.lower() or 'collection' in src.lower() or 
                           'look' in src.lower() or 'slide' in src.lower()):
                    # Get highest resolution
                    if 'w_' in src:
                        src = re.sub(r'w_\d+', 'w_1200', src)
                    
                    looks.append({
                        'image_url': src,
                        'alt': img.get('alt', '')
                    })
            
        except Exception as e:
            logger.warning(f"Error getting looks from {designer_url}: {e}")
        
        return looks
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """Harvest runway looks from Vogue."""
        
        for season in self.SEASONS:
            logger.info(f"Harvesting Vogue Runway: {season}")
            
            designers = self._get_season_designers(season)
            logger.info(f"Found {len(designers)} designers for {season}")
            
            for designer in designers:
                logger.info(f"Processing: {designer['name']}")
                
                looks = self._get_collection_looks(designer['url'])
                
                for i, look in enumerate(looks):
                    try:
                        img_url = look['image_url']
                        local_path = self.download_image(img_url)
                        
                        item = FashionItem(
                            source=self.name,
                            source_url=designer['url'],
                            image_url=img_url,
                            local_path=local_path,
                            title=f"{designer['name']} - Look {i+1}",
                            content=look.get('alt', ''),
                            tags=[season.replace('-', ' '), 'runway', 'designer'],
                            category='runway',
                            metadata={
                                'designer': designer['name'],
                                'season': season,
                                'look_number': i + 1
                            }
                        )
                        
                        self.items_harvested += 1
                        yield item
                        
                    except Exception as e:
                        logger.warning(f"Error processing look: {e}")
                        self.errors += 1
                
                self.rate_limit(2.0, 4.0)
                self.rotate_user_agent()


class TheImpressionHarvester(FashionSource):
    """
    Harvests runway imagery from The Impression.
    
    Alternative to Vogue Runway with different angle
    on fashion week coverage.
    """
    
    BASE_URL = "https://theimpression.com"
    
    def __init__(self, data_dir: str = "./data/harvested"):
        super().__init__(data_dir)
        
    @property
    def name(self) -> str:
        return "the_impression"
    
    def validate(self) -> bool:
        """Validate The Impression access."""
        if not BS4_AVAILABLE:
            return False
        
        try:
            response = self.session.get(self.BASE_URL, timeout=15)
            return response.status_code == 200
        except Exception:
            return False
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """Harvest from The Impression."""
        
        try:
            # Get runway section
            runway_url = f"{self.BASE_URL}/runway/"
            response = self.session.get(runway_url, timeout=15)
            
            if response.status_code != 200:
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find collection links
            articles = soup.find_all('article')[:20]
            
            for article in articles:
                try:
                    link = article.find('a')
                    if not link:
                        continue
                    
                    collection_url = link.get('href')
                    title = link.get_text(strip=True)[:100]
                    
                    # Find image
                    img = article.find('img')
                    if img:
                        img_url = img.get('src') or img.get('data-src')
                        
                        if img_url:
                            local_path = self.download_image(img_url)
                            
                            item = FashionItem(
                                source=self.name,
                                source_url=collection_url or self.BASE_URL,
                                image_url=img_url,
                                local_path=local_path,
                                title=title,
                                tags=['runway', 'fashion week'],
                                category='runway'
                            )
                            
                            self.items_harvested += 1
                            yield item
                    
                    self.rate_limit(1.0, 2.0)
                    
                except Exception as e:
                    logger.warning(f"Error processing article: {e}")
                    
        except Exception as e:
            logger.error(f"Error harvesting The Impression: {e}")


__all__ = ['VogueRunwayHarvester', 'TheImpressionHarvester']
