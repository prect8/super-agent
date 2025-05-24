import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import re
import json
from bs4 import BeautifulSoup
import wikipedia
from duckduckgo_search import DDGS
import arxiv
import feedparser
from urllib.parse import urljoin, urlparse

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FreeResearchEngine:
    """無料でディープリサーチを行うエンジン"""
    
    def __init__(self):
        self.max_results_per_source = 5
        self.timeout = 30
        
        # ニュースRSSフィード（無料）
        self.news_feeds = [
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://www.reuters.com/rssFeed/topNews",
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
        ]
    
    def extract_keywords(self, query: str) -> List[str]:
        """クエリからキーワードを抽出"""
        # 基本的なキーワード抽出
        keywords = []
        
        # 日本語と英語の単語を抽出
        words = re.findall(r'\b\w+\b', query.lower())
        
        # ストップワードを除去
        stop_words = {'の', 'は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 元のクエリも追加
        keywords.insert(0, query)
        
        return keywords[:5]  # 上位5つのキーワード
    
    async def search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """DuckDuckGo検索（レート制限対策付き）"""
        try:
            import time
            import random
            
            # レート制限回避のための遅延
            await asyncio.sleep(random.uniform(1, 3))
            
            ddgs = DDGS()
            results = []
            
            # ウェブ検索（エラーハンドリング強化）
            try:
                search_results = ddgs.text(query, max_results=self.max_results_per_source)
                for result in search_results:
                    results.append({
                        "source": "DuckDuckGo Web",
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.warning(f"DuckDuckGo web search failed: {e}")
            
            # レート制限回避のための追加遅延
            await asyncio.sleep(random.uniform(2, 4))
            
            # ニュース検索（エラーハンドリング強化）
            try:
                news_results = ddgs.news(query, max_results=3)
                for result in news_results:
                    results.append({
                        "source": "DuckDuckGo News",
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("body", ""),
                        "timestamp": result.get("date", datetime.now().isoformat())
                    })
            except Exception as e:
                logger.warning(f"DuckDuckGo news search failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    async def search_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        """Wikipedia検索（無料）"""
        try:
            results = []
            
            # 日本語と英語で検索
            for lang in ['ja', 'en']:
                wikipedia.set_lang(lang)
                
                try:
                    # 検索候補を取得
                    search_results = wikipedia.search(query, results=3)
                    
                    for title in search_results:
                        try:
                            page = wikipedia.page(title)
                            results.append({
                                "source": f"Wikipedia ({lang})",
                                "title": page.title,
                                "url": page.url,
                                "snippet": page.summary[:500] + "...",
                                "content": page.content[:2000] + "...",
                                "timestamp": datetime.now().isoformat()
                            })
                        except wikipedia.exceptions.DisambiguationError as e:
                            # 曖昧さ回避ページの場合、最初の選択肢を使用
                            try:
                                page = wikipedia.page(e.options[0])
                                results.append({
                                    "source": f"Wikipedia ({lang})",
                                    "title": page.title,
                                    "url": page.url,
                                    "snippet": page.summary[:500] + "...",
                                    "content": page.content[:2000] + "...",
                                    "timestamp": datetime.now().isoformat()
                                })
                            except:
                                continue
                        except:
                            continue
                            
                except Exception as e:
                    logger.error(f"Wikipedia search error for {lang}: {e}")
                    continue
            
            return results[:self.max_results_per_source]
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    async def search_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """ArXiv学術論文検索（無料）"""
        try:
            results = []
            
            # ArXiv検索
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                results.append({
                    "source": "ArXiv",
                    "title": result.title,
                    "url": result.entry_id,
                    "snippet": result.summary[:500] + "...",
                    "authors": [author.name for author in result.authors],
                    "published": result.published.isoformat(),
                    "timestamp": datetime.now().isoformat()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []
    
    async def search_news_feeds(self, query: str) -> List[Dict[str, Any]]:
        """ニュースRSSフィード検索（無料）"""
        try:
            results = []
            keywords = self.extract_keywords(query)
            
            for feed_url in self.news_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:5]:  # 各フィードから5件
                        # キーワードマッチング
                        title_lower = entry.title.lower()
                        summary_lower = getattr(entry, 'summary', '').lower()
                        
                        if any(keyword.lower() in title_lower or keyword.lower() in summary_lower 
                               for keyword in keywords):
                            results.append({
                                "source": f"News Feed ({urlparse(feed_url).netloc})",
                                "title": entry.title,
                                "url": entry.link,
                                "snippet": getattr(entry, 'summary', '')[:500] + "...",
                                "published": getattr(entry, 'published', ''),
                                "timestamp": datetime.now().isoformat()
                            })
                            
                except Exception as e:
                    logger.error(f"News feed error for {feed_url}: {e}")
                    continue
            
            return results[:self.max_results_per_source]
            
        except Exception as e:
            logger.error(f"News feeds search error: {e}")
            return []
    
    async def scrape_content(self, url: str) -> str:
        """Webページのコンテンツを取得（無料）"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # 不要なタグを除去
                        for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
                            tag.decompose()
                        
                        # メインコンテンツを抽出
                        text = soup.get_text()
                        lines = [line.strip() for line in text.splitlines()]
                        content = ' '.join(line for line in lines if line)
                        
                        return content[:3000] + "..."  # 3000文字まで
                    
        except Exception as e:
            logger.error(f"Content scraping error for {url}: {e}")
            
        return ""
    
    async def conduct_deep_research(self, query: str) -> Dict[str, Any]:
        """ディープリサーチの実行"""
        research_start = datetime.now()
        
        logger.info(f"Starting deep research for: {query}")
        
        # 並行して複数のソースから検索
        search_tasks = [
            self.search_duckduckgo(query),
            self.search_wikipedia(query),
            self.search_arxiv(query),
            self.search_news_feeds(query)
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # 結果をマージ
        all_results = []
        for result_set in search_results:
            if isinstance(result_set, list):
                all_results.extend(result_set)
        
        # 重要なページのコンテンツを詳細取得
        content_tasks = []
        for result in all_results[:5]:  # 上位5件の詳細コンテンツを取得
            if result.get("url"):
                content_tasks.append(self.scrape_content(result["url"]))
        
        detailed_contents = await asyncio.gather(*content_tasks, return_exceptions=True)
        
        # 詳細コンテンツをマージ
        for i, content in enumerate(detailed_contents):
            if isinstance(content, str) and content and i < len(all_results):
                all_results[i]["detailed_content"] = content
        
        research_duration = (datetime.now() - research_start).total_seconds()
        
        return {
            "query": query,
            "total_results": len(all_results),
            "research_duration": research_duration,
            "sources_used": list(set(result["source"] for result in all_results)),
            "results": all_results,
            "timestamp": research_start.isoformat()
        }
    
    def generate_research_summary(self, research_data: Dict[str, Any]) -> str:
        """リサーチ結果のサマリーを生成"""
        query = research_data["query"]
        results = research_data["results"]
        sources = research_data["sources_used"]
        
        summary = f"# 📊 '{query}' のディープリサーチ結果\n\n"
        summary += f"**検索実行時刻**: {research_data['timestamp']}\n"
        summary += f"**検索時間**: {research_data['research_duration']:.2f}秒\n"
        summary += f"**情報源**: {', '.join(sources)}\n"
        summary += f"**収集した情報数**: {research_data['total_results']}件\n\n"
        
        summary += "## 🔍 主要な発見\n\n"
        
        # カテゴリ別に整理
        by_source = {}
        for result in results:
            source = result["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)
        
        for source, source_results in by_source.items():
            summary += f"### {source}\n\n"
            for result in source_results[:3]:  # 各ソースから上位3件
                summary += f"**{result['title']}**\n"
                summary += f"{result['snippet']}\n"
                if result.get('url'):
                    summary += f"🔗 [詳細]({result['url']})\n\n"
        
        return summary

# 使用例
async def main():
    engine = FreeResearchEngine()
    query = "人工知能の最新動向"
    research_data = await engine.conduct_deep_research(query)
    summary = engine.generate_research_summary(research_data)
    print(summary)

if __name__ == "__main__":
    asyncio.run(main()) 