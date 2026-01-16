"""
opentrend/harvesters/street_signal.py
"Street Signal" Harvester - Social Media Sources

Captures what real people are discussing and wearing RIGHT NOW.
Sources: Reddit (r/streetwear, r/femalefashionadvice) + Pinterest RSS

This harvester focuses on grassroots fashion signals before they
hit mainstream trend reports.
"""

from __future__ import annotations

import re
import time
import random
import logging
from typing import Generator, Optional, List
from datetime import datetime

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

from .base import FashionSource, FashionItem

logger = logging.getLogger(__name__)


class RedditHarvester(FashionSource):
    """
    Harvests fashion content from Reddit fashion communities.
    
    Subreddits:
    - r/streetwear - Street fashion trends
    - r/femalefashionadvice - Women's fashion discussions
    - r/malefashionadvice - Men's fashion discussions
    - r/techwear - Technical/futuristic fashion
    - r/fashionreps - Replica discussions (for trend signals)
    
    Requires Reddit API credentials (free tier works).
    """
    
    # Fashion subreddits to monitor
    SUBREDDITS = [
        'streetwear',
        'femalefashionadvice', 
        'malefashionadvice',
        'techwear',
        'fashionreps',
        'womensstreetwear',
        'sneakers',
        'rawdenim'
    ]
    
    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        user_agent: str = "OpenTrend Fashion Brain 1.0",
        posts_per_subreddit: int = 50,
        data_dir: str = "./data/harvested"
    ):
        super().__init__(data_dir)
        
        self.client_id = client_id
        self.client_secret = client_secret  
        self.user_agent = user_agent
        self.posts_per_sub = posts_per_subreddit
        self.reddit = None
        
    @property
    def name(self) -> str:
        return "reddit"
    
    def validate(self) -> bool:
        """Check Reddit API availability."""
        if not PRAW_AVAILABLE:
            logger.warning("praw not installed. Run: pip install praw")
            return False
        
        # If no credentials, use read-only mode
        try:
            if self.client_id and self.client_secret:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
            else:
                # Read-only mode without auth
                self.reddit = praw.Reddit(
                    client_id='dummy',
                    client_secret='dummy',
                    user_agent=self.user_agent
                )
                self.reddit.read_only = True
            return True
        except Exception as e:
            logger.error(f"Reddit validation failed: {e}")
            return False
    
    def _extract_image_urls(self, submission) -> List[str]:
        """Extract image URLs from a Reddit submission."""
        urls = []
        
        # Direct image link
        if hasattr(submission, 'url'):
            url = submission.url
            if any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                urls.append(url)
            elif 'imgur.com' in url and not url.endswith('.gifv'):
                # Convert imgur to direct link
                if not url.endswith('.jpg'):
                    urls.append(f"{url}.jpg")
        
        # Gallery posts
        if hasattr(submission, 'is_gallery') and submission.is_gallery:
            if hasattr(submission, 'media_metadata'):
                for item in submission.media_metadata.values():
                    if 's' in item and 'u' in item['s']:
                        urls.append(item['s']['u'].replace('&amp;', '&'))
        
        # Preview images
        if hasattr(submission, 'preview') and 'images' in submission.preview:
            for img in submission.preview['images']:
                if 'source' in img:
                    urls.append(img['source']['url'].replace('&amp;', '&'))
        
        return urls[:5]  # Limit to 5 images per post
    
    def _extract_fashion_terms(self, text: str) -> List[str]:
        """Extract fashion-related terms from text."""
        fashion_keywords = [
            'outfit', 'fit', 'drip', 'aesthetic', 'vintage', 'thrift',
            'streetwear', 'minimalist', 'oversized', 'layering', 'monochrome',
            'denim', 'leather', 'sneakers', 'boots', 'jacket', 'hoodie',
            'pants', 'jeans', 'shirt', 'blazer', 'coat', 'dress',
            'y2k', 'gorpcore', 'techwear', 'normcore', 'cottagecore',
            'cyberpunk', 'boho', 'grunge', 'preppy', 'athleisure'
        ]
        
        text_lower = text.lower()
        found = [kw for kw in fashion_keywords if kw in text_lower]
        return found
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """Harvest fashion content from Reddit."""
        if not self.reddit:
            return
        
        for subreddit_name in self.SUBREDDITS:
            try:
                logger.info(f"Harvesting r/{subreddit_name}")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot and new posts
                for submission in subreddit.hot(limit=self.posts_per_sub):
                    try:
                        # Skip non-image posts for now
                        image_urls = self._extract_image_urls(submission)
                        
                        # Extract fashion terms
                        full_text = f"{submission.title} {submission.selftext}"
                        tags = self._extract_fashion_terms(full_text)
                        
                        # Create item for each image
                        for img_url in image_urls:
                            # Download image
                            local_path = self.download_image(img_url)
                            
                            item = FashionItem(
                                source=self.name,
                                source_url=f"https://reddit.com{submission.permalink}",
                                image_url=img_url,
                                local_path=local_path,
                                title=submission.title[:200],
                                content=submission.selftext[:500] if submission.selftext else "",
                                tags=tags,
                                category=subreddit_name,
                                engagement=submission.score,
                                metadata={
                                    'subreddit': subreddit_name,
                                    'upvote_ratio': submission.upvote_ratio,
                                    'num_comments': submission.num_comments,
                                    'created_utc': submission.created_utc
                                }
                            )
                            
                            self.items_harvested += 1
                            yield item
                        
                        self.rate_limit(0.5, 1.5)
                        
                    except Exception as e:
                        logger.warning(f"Error processing submission: {e}")
                        self.errors += 1
                        continue
                
            except Exception as e:
                logger.error(f"Error harvesting r/{subreddit_name}: {e}")
                self.errors += 1
                continue


class PinterestHarvester(FashionSource):
    """
    Harvests fashion imagery from Pinterest.
    
    Uses RSS feeds and public search pages to extract
    trending fashion pins without requiring API access.
    
    Note: This uses ethical scraping of public content.
    """
    
    # Fashion-related search terms
    SEARCH_TERMS = [
        'street style 2026',
        'fashion week looks',
        'outfit inspo aesthetic',
        'avant garde fashion',
        'minimalist outfit',
        'y2k fashion',
        'techwear outfit',
        'boho chic',
        'sustainable fashion'
    ]
    
    # RSS feed URLs for fashion categories  
    RSS_FEEDS = [
        'https://www.pinterest.com/fashionweekdaily.xml',
    ]
    
    def __init__(self, data_dir: str = "./data/harvested"):
        super().__init__(data_dir)
        
    @property
    def name(self) -> str:
        return "pinterest"
    
    def validate(self) -> bool:
        """Validate Pinterest access."""
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not installed. Run: pip install feedparser")
        
        # Test basic HTTP access
        try:
            response = self.session.get(
                "https://www.pinterest.com",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Pinterest validation failed: {e}")
            return False
    
    def _parse_pinterest_page(self, search_term: str) -> List[dict]:
        """Parse Pinterest search page for pins."""
        results = []
        
        try:
            # Pinterest search URL
            search_url = f"https://www.pinterest.com/search/pins/?q={search_term.replace(' ', '%20')}"
            
            response = self.session.get(search_url, timeout=15)
            
            if response.status_code == 200:
                # Extract image URLs from page content
                # Pinterest loads dynamically, so we get what's in initial HTML
                import re
                
                # Find image URLs
                img_pattern = r'https://i\.pinimg\.com/[^"\'>\s]+'
                images = re.findall(img_pattern, response.text)
                
                # Filter for high-quality originals
                for img_url in set(images[:30]):  # Limit per search
                    if '/originals/' in img_url or '/564x/' in img_url:
                        results.append({
                            'image_url': img_url,
                            'search_term': search_term
                        })
            
        except Exception as e:
            logger.warning(f"Error parsing Pinterest for '{search_term}': {e}")
        
        return results
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """Harvest fashion pins from Pinterest."""
        
        for search_term in self.SEARCH_TERMS:
            logger.info(f"Harvesting Pinterest: '{search_term}'")
            
            pins = self._parse_pinterest_page(search_term)
            
            for pin in pins:
                try:
                    img_url = pin['image_url']
                    local_path = self.download_image(img_url)
                    
                    item = FashionItem(
                        source=self.name,
                        source_url=img_url,
                        image_url=img_url,
                        local_path=local_path,
                        title=pin.get('search_term', ''),
                        tags=[t.strip() for t in search_term.split()],
                        category='pinterest_search',
                        metadata={'search_term': search_term}
                    )
                    
                    self.items_harvested += 1
                    yield item
                    
                except Exception as e:
                    logger.warning(f"Error processing pin: {e}")
                    self.errors += 1
            
            self.rate_limit(2.0, 5.0)  # Be respectful to Pinterest
            self.rotate_user_agent()


# Convenience export
__all__ = ['RedditHarvester', 'PinterestHarvester']
