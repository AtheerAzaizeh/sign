"""
opentrend/harvesters/visual_vibe.py
"Visual Vibe" Harvester - Open Source Image APIs

High-quality, royalty-free fashion imagery from Unsplash and Pexels.
Perfect for training data as images are legally usable.

Targets:
- Street style photography
- OOTD (Outfit of the Day)
- Avant-garde fashion
- Sustainable fashion
"""

from __future__ import annotations

import os
import logging
from typing import Generator, List
from datetime import datetime

from .base import FashionSource, FashionItem

logger = logging.getLogger(__name__)


class UnsplashHarvester(FashionSource):
    """
    Harvests fashion imagery from Unsplash API.
    
    Requires free API key from: https://unsplash.com/developers
    
    Unsplash provides high-quality, professionally photographed
    images with permissive licensing for AI training.
    """
    
    API_BASE = "https://api.unsplash.com"
    
    # Fashion-related search queries
    SEARCH_QUERIES = [
        'street style fashion',
        'outfit of the day',
        'fashion model',
        'avant garde fashion',
        'minimalist fashion',
        'sustainable fashion',
        'vintage clothing',
        'streetwear',
        'fashion week',
        'designer fashion',
        'urban style',
        'bohemian fashion',
        'techwear',
        'athleisure'
    ]
    
    def __init__(
        self,
        api_key: str = None,
        images_per_query: int = 30,
        data_dir: str = "./data/harvested"
    ):
        super().__init__(data_dir)
        self.api_key = api_key or os.environ.get('UNSPLASH_API_KEY')
        self.images_per_query = images_per_query
        
    @property
    def name(self) -> str:
        return "unsplash"
    
    def validate(self) -> bool:
        """Validate Unsplash API access."""
        if not self.api_key:
            logger.warning("UNSPLASH_API_KEY not set")
            # Can still work without key (limited requests)
            return True
        
        try:
            headers = {'Authorization': f'Client-ID {self.api_key}'}
            response = self.session.get(
                f"{self.API_BASE}/photos/random",
                headers=headers,
                timeout=10
            )
            return response.status_code in [200, 401]  # 401 = invalid key but service works
        except Exception as e:
            logger.warning(f"Unsplash validation failed: {e}")
            return False
    
    def _search_photos(self, query: str, page: int = 1) -> List[dict]:
        """Search Unsplash for photos."""
        photos = []
        
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Client-ID {self.api_key}'
            
            params = {
                'query': query,
                'per_page': min(30, self.images_per_query),
                'page': page,
                'orientation': 'portrait'  # Fashion photos are often portrait
            }
            
            response = self.session.get(
                f"{self.API_BASE}/search/photos",
                headers=headers,
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for photo in data.get('results', []):
                    photos.append({
                        'id': photo['id'],
                        'image_url': photo['urls'].get('regular') or photo['urls'].get('small'),
                        'full_url': photo['urls'].get('full'),
                        'description': photo.get('description') or photo.get('alt_description', ''),
                        'tags': [t.get('title', '') for t in photo.get('tags', [])],
                        'likes': photo.get('likes', 0),
                        'user': photo.get('user', {}).get('username', ''),
                        'download_url': photo.get('links', {}).get('download')
                    })
            elif response.status_code == 403:
                logger.warning("Unsplash rate limit reached")
            
        except Exception as e:
            logger.warning(f"Error searching Unsplash for '{query}': {e}")
        
        return photos
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """Harvest fashion photos from Unsplash."""
        
        for query in self.SEARCH_QUERIES:
            logger.info(f"Harvesting Unsplash: '{query}'")
            
            photos = self._search_photos(query)
            
            for photo in photos:
                try:
                    img_url = photo['image_url']
                    local_path = self.download_image(img_url, f"{photo['id']}.jpg")
                    
                    item = FashionItem(
                        source=self.name,
                        source_url=f"https://unsplash.com/photos/{photo['id']}",
                        image_url=img_url,
                        local_path=local_path,
                        title=photo['description'][:200] if photo['description'] else query,
                        tags=photo['tags'][:10],
                        category=query.replace(' ', '_'),
                        engagement=photo['likes'],
                        metadata={
                            'photographer': photo['user'],
                            'unsplash_id': photo['id'],
                            'query': query
                        }
                    )
                    
                    self.items_harvested += 1
                    yield item
                    
                except Exception as e:
                    logger.warning(f"Error processing Unsplash photo: {e}")
                    self.errors += 1
            
            self.rate_limit(1.0, 2.0)


class PexelsHarvester(FashionSource):
    """
    Harvests fashion imagery from Pexels API.
    
    Requires free API key from: https://www.pexels.com/api/
    
    Pexels provides diverse, high-quality stock photography
    with broad representation.
    """
    
    API_BASE = "https://api.pexels.com/v1"
    
    SEARCH_QUERIES = [
        'street fashion',
        'fashion portrait',
        'model outfit',
        'casual wear',
        'formal fashion',
        'urban fashion',
        'fashion accessories',
        'summer fashion',
        'winter fashion',
        'sustainable clothing'
    ]
    
    def __init__(
        self,
        api_key: str = None,
        images_per_query: int = 30,
        data_dir: str = "./data/harvested"
    ):
        super().__init__(data_dir)
        self.api_key = api_key or os.environ.get('PEXELS_API_KEY')
        self.images_per_query = images_per_query
        
    @property
    def name(self) -> str:
        return "pexels"
    
    def validate(self) -> bool:
        """Validate Pexels API access."""
        if not self.api_key:
            logger.warning("PEXELS_API_KEY not set. Skipping Pexels harvester.")
            return False
        
        try:
            headers = {'Authorization': self.api_key}
            response = self.session.get(
                f"{self.API_BASE}/curated",
                headers=headers,
                params={'per_page': 1},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Pexels validation failed: {e}")
            return False
    
    def _search_photos(self, query: str, page: int = 1) -> List[dict]:
        """Search Pexels for photos."""
        photos = []
        
        try:
            headers = {'Authorization': self.api_key}
            params = {
                'query': query,
                'per_page': min(80, self.images_per_query),
                'page': page,
                'orientation': 'portrait'
            }
            
            response = self.session.get(
                f"{self.API_BASE}/search",
                headers=headers,
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for photo in data.get('photos', []):
                    photos.append({
                        'id': photo['id'],
                        'image_url': photo['src'].get('large') or photo['src'].get('medium'),
                        'original_url': photo['src'].get('original'),
                        'description': photo.get('alt', ''),
                        'photographer': photo.get('photographer', ''),
                        'url': photo.get('url', '')
                    })
                    
        except Exception as e:
            logger.warning(f"Error searching Pexels for '{query}': {e}")
        
        return photos
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """Harvest fashion photos from Pexels."""
        
        for query in self.SEARCH_QUERIES:
            logger.info(f"Harvesting Pexels: '{query}'")
            
            photos = self._search_photos(query)
            
            for photo in photos:
                try:
                    img_url = photo['image_url']
                    local_path = self.download_image(img_url, f"{photo['id']}.jpg")
                    
                    item = FashionItem(
                        source=self.name,
                        source_url=photo['url'],
                        image_url=img_url,
                        local_path=local_path,
                        title=photo['description'][:200] if photo['description'] else query,
                        tags=[t.strip() for t in query.split()],
                        category=query.replace(' ', '_'),
                        metadata={
                            'photographer': photo['photographer'],
                            'pexels_id': photo['id'],
                            'query': query
                        }
                    )
                    
                    self.items_harvested += 1
                    yield item
                    
                except Exception as e:
                    logger.warning(f"Error processing Pexels photo: {e}")
                    self.errors += 1
            
            self.rate_limit(0.5, 1.5)  # Pexels allows more requests


__all__ = ['UnsplashHarvester', 'PexelsHarvester']
