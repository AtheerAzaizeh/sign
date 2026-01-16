"""
opentrend/harvesters/marketplace.py
"Hype" Harvester - Marketplace Sources

Monitors resale marketplaces to understand immediate demand.
Sources: Depop, Grailed, TheRealReal (public listings)

Key signal: Items that sell quickly indicate urgent demand.
"""

from __future__ import annotations

import re
import logging
from typing import Generator, List
from datetime import datetime

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from .base import FashionSource, FashionItem

logger = logging.getLogger(__name__)


class DepopHarvester(FashionSource):
    """
    Harvests listings from Depop.
    
    Depop is popular with Gen-Z for vintage, Y2K, and
    streetwear. Fast-selling items indicate strong demand.
    
    Scrapes:
    - Trending items
    - Recently sold items
    - Popular category listings
    """
    
    BASE_URL = "https://www.depop.com"
    
    # Categories to monitor
    CATEGORIES = [
        'womens/clothing/tops',
        'womens/clothing/dresses',
        'womens/clothing/jackets-coats',
        'mens/clothing/t-shirts',
        'mens/clothing/hoodies-sweatshirts',
        'mens/clothing/shirts',
        'womens/shoes/sneakers',
        'mens/shoes/sneakers'
    ]
    
    # Trending search terms
    TRENDING_SEARCHES = [
        'y2k',
        'vintage',
        'streetwear',
        'coquette',
        'balletcore',
        'grunge',
        'fairycore',
        'mob wife'
    ]
    
    def __init__(
        self,
        items_per_category: int = 50,
        data_dir: str = "./data/harvested"
    ):
        super().__init__(data_dir)
        self.items_per_category = items_per_category
        
    @property
    def name(self) -> str:
        return "depop"
    
    def validate(self) -> bool:
        """Validate Depop access."""
        if not BS4_AVAILABLE:
            logger.warning("beautifulsoup4 not installed")
            return False
        
        try:
            response = self.session.get(self.BASE_URL, timeout=15)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Depop validation failed: {e}")
            return False
    
    def _parse_search_page(self, search_term: str) -> List[dict]:
        """Parse Depop search results."""
        items = []
        
        try:
            url = f"{self.BASE_URL}/search/?q={search_term.replace(' ', '%20')}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return items
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find product cards
            products = soup.find_all('a', href=re.compile(r'/products/'))[:self.items_per_category]
            
            for product in products:
                try:
                    img = product.find('img')
                    img_url = img.get('src') if img else None
                    
                    title_elem = product.find(class_=re.compile('title|name|description'))
                    title = title_elem.get_text(strip=True) if title_elem else ''
                    
                    price_elem = product.find(class_=re.compile('price'))
                    price = price_elem.get_text(strip=True) if price_elem else ''
                    
                    href = product.get('href', '')
                    product_url = f"{self.BASE_URL}{href}" if not href.startswith('http') else href
                    
                    if img_url:
                        items.append({
                            'image_url': img_url,
                            'title': title[:200],
                            'price': price,
                            'url': product_url,
                            'search_term': search_term
                        })
                        
                except Exception as e:
                    continue
            
        except Exception as e:
            logger.warning(f"Error parsing Depop for '{search_term}': {e}")
        
        return items
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """Harvest listings from Depop."""
        
        # Harvest trending searches
        for search_term in self.TRENDING_SEARCHES:
            logger.info(f"Harvesting Depop: '{search_term}'")
            
            items = self._parse_search_page(search_term)
            
            for item_data in items:
                try:
                    img_url = item_data['image_url']
                    local_path = self.download_image(img_url)
                    
                    item = FashionItem(
                        source=self.name,
                        source_url=item_data['url'],
                        image_url=img_url,
                        local_path=local_path,
                        title=item_data['title'],
                        tags=[search_term, 'resale', 'depop'],
                        category='marketplace',
                        metadata={
                            'price': item_data['price'],
                            'search_term': search_term,
                            'platform': 'depop'
                        }
                    )
                    
                    self.items_harvested += 1
                    yield item
                    
                except Exception as e:
                    logger.warning(f"Error processing Depop item: {e}")
                    self.errors += 1
            
            self.rate_limit(2.0, 4.0)
            self.rotate_user_agent()


class GrailedHarvester(FashionSource):
    """
    Harvests listings from Grailed.
    
    Grailed specializes in menswear, designer, and streetwear.
    Great source for:
    - High-end streetwear trends
    - Designer resale signals
    - Rare/grailed pieces
    """
    
    BASE_URL = "https://www.grailed.com"
    
    # Grailed categories
    CATEGORIES = [
        'designers/supreme',
        'designers/nike',
        'designers/rick-owens',
        'designers/raf-simons',
        'designers/comme-des-garcons',
        'designers/acne-studios',
        'categories/tops',
        'categories/outerwear'
    ]
    
    def __init__(
        self,
        items_per_category: int = 30,
        data_dir: str = "./data/harvested"
    ):
        super().__init__(data_dir)
        self.items_per_category = items_per_category
        
    @property
    def name(self) -> str:
        return "grailed"
    
    def validate(self) -> bool:
        """Validate Grailed access."""
        if not BS4_AVAILABLE:
            return False
        
        try:
            response = self.session.get(self.BASE_URL, timeout=15)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_feed_items(self) -> List[dict]:
        """Get items from Grailed feed."""
        items = []
        
        try:
            # Grailed feed API (public)
            feed_url = f"{self.BASE_URL}/api/listings/grailed"
            
            response = self.session.get(
                feed_url,
                headers={
                    **self.session.headers,
                    'Accept': 'application/json'
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for listing in data.get('data', [])[:self.items_per_category]:
                    photos = listing.get('photos', [])
                    
                    if photos:
                        items.append({
                            'id': listing.get('id'),
                            'image_url': photos[0].get('url') if photos else None,
                            'title': listing.get('title', ''),
                            'designer': listing.get('designer', {}).get('name', ''),
                            'price': listing.get('price'),
                            'category': listing.get('category', ''),
                            'sold': listing.get('sold', False)
                        })
                        
        except Exception as e:
            logger.warning(f"Error getting Grailed feed: {e}")
        
        return items
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """Harvest listings from Grailed."""
        logger.info("Harvesting Grailed feed")
        
        items = self._get_feed_items()
        
        for item_data in items:
            try:
                img_url = item_data.get('image_url')
                if not img_url:
                    continue
                    
                local_path = self.download_image(img_url)
                
                item = FashionItem(
                    source=self.name,
                    source_url=f"{self.BASE_URL}/listings/{item_data['id']}",
                    image_url=img_url,
                    local_path=local_path,
                    title=item_data['title'][:200],
                    tags=[item_data.get('designer', ''), item_data.get('category', ''), 'grailed'],
                    category='marketplace',
                    metadata={
                        'designer': item_data.get('designer'),
                        'price': item_data.get('price'),
                        'sold': item_data.get('sold'),
                        'grailed_id': item_data.get('id')
                    }
                )
                
                self.items_harvested += 1
                yield item
                
            except Exception as e:
                logger.warning(f"Error processing Grailed item: {e}")
                self.errors += 1
        
        self.rate_limit(2.0, 4.0)


__all__ = ['DepopHarvester', 'GrailedHarvester']
