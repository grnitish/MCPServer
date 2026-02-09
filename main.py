"""
Serper API News Fetcher
Fetches U.S. news and stores as JSON files
"""
import requests
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()   
import trafilatura
import time


class SerperNewsFetcher:
    """Fetch news using Serper API"""
    
    def __init__(self, api_key: str, output_folder: str = "news_data"):
        self.api_key = api_key
        self.output_folder = output_folder
        self.base_url = "https://google.serper.dev/news"
        
        # Create output folder if doesn't exist
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ Output folder: {self.output_folder}")
    
    def fetch_news(self, query: str = "USA news", num_results: int = 100):
        """
        Fetch news from Serper API
        
        Args:
            query: Search query (default: "USA news")
            num_results: Number of results (max 100)
        
        Returns:
            List of news articles
        """
        print(f"🔍 Fetching news: '{query}' (max {num_results} results)")
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            "q": query,
            "num": num_results,
            "tbs": "qdr:d"  # Last 24 hours (d=day, w=week, m=month)
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            articles = data.get('news', [])
            print(f"✅ Fetched {len(articles)} articles")
            print(articles)
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching news: {e}")
            return []
    
    def enrich_articles_with_content(self, articles: list, delay: float = 1.0):
        """
        Fetch full article content for each news item

        Args:
            articles: List of article dictionaries
            delay: Seconds to wait between requests (polite crawling)

        Returns:
            List of enriched articles
        """
        print("🧠 Fetching full article content...")

        for i, article in enumerate(articles, 1):
            url = article.get("link")
            if not url:
                article["content"] = None
                continue

            try:
                downloaded = trafilatura.fetch_url(url)
                content = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False
                )

                article["content"] = content
                article["content_length"] = len(content) if content else 0

                print(f"  [{i}/{len(articles)}] ✓ Content fetched")

            except Exception as e:
                article["content"] = None
                article["error"] = str(e)
                print(f"  [{i}/{len(articles)}] ✗ Failed")

            time.sleep(delay)  # avoid hammering sites

        return articles

    def save_articles(self, articles: list, filename: str = None):
        """
        Save articles to JSON file
        
        Args:
            articles: List of article dictionaries
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if not articles:
            print("⚠️ No articles to save")
            return None
        
        # Generate filename with timestamp
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_{timestamp}.json"
        
        filepath = os.path.join(self.output_folder, filename)
        
        # Prepare data with metadata
        data = {
            "fetched_at": datetime.now().isoformat(),
            "total_articles": len(articles),
            "articles": articles
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(articles)} articles to: {filepath}")
        return filepath
    
    def fetch_and_save(self, query: str = "USA news", num_results: int = 100):
        """
        Fetch news and save in one step
        
        Args:
            query: Search query
            num_results: Number of results
        
        Returns:
            Path to saved file
        """
        # Fetch
        articles = self.fetch_news(query, num_results)
        #articles = self.fetch_news(query, num_results)
        
        # Save
        if articles:
            articles = self.enrich_articles_with_content(articles)
            filepath = self.save_articles(articles)

            return filepath
        
        return None
    
    def display_articles(self, articles: list, limit: int = 5):
        """Display articles in terminal"""
        print(f"\n📰 Showing {min(limit, len(articles))} articles:\n")
        
        for i, article in enumerate(articles[:limit], 1):
            print(f"{i}. {article.get('title', 'No title')}")
            print(f"   Source: {article.get('source', 'Unknown')}")
            print(f"   Date: {article.get('date', 'Unknown')}")
            print(f"   Link: {article.get('link', 'No link')[:60]}...")
            print()


def main():
    """Main function"""
    
    # ========================================
    # CONFIGURATION - ADD YOUR API KEY HERE
    # ========================================
    SERPER_API_KEY = os.environ.get("SERPER_API")  # ← CHANGE THIS!
    #print(SERPER_API_KEY,'hi')
    # Check if API key is set
    if SERPER_API_KEY == "":
        print("❌ ERROR: Please set your Serper API key!")
        print("Get it from: https://serper.dev/")
        return
    
    # ========================================
    # FETCH NEWS
    # ========================================
    
    print("=" * 60)
    print("📰 SERPER NEWS FETCHER")
    print("=" * 60)
    print()
    
    # Initialize fetcher
    fetcher = SerperNewsFetcher(
        api_key=SERPER_API_KEY,
        output_folder="news_data"
    )
    
    # Fetch and save news
    filepath = fetcher.fetch_and_save(
        query="telsa car news",
        num_results=100
    )
    
    if filepath:
        # Load and display
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fetcher.display_articles(data['articles'], limit=10)
        
        print("=" * 60)
        print(f"✅ SUCCESS!")
        print(f"📁 File saved: {filepath}")
        print(f"📊 Total articles: {data['total_articles']}")
        print("=" * 60)


if __name__ == "__main__":
    main()