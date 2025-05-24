import asyncio
from research import FreeResearchEngine

async def test_research_functionality():
    """リサーチ機能の詳細テスト"""
    print("=== Research Functionality Test ===")
    
    try:
        engine = FreeResearchEngine()
        print("✅ Research engine created successfully")
        
        # 簡単なリサーチテスト
        test_query = "Python programming"
        print(f"Testing research for: '{test_query}'")
        
        result = await engine.conduct_deep_research(test_query)
        
        print(f"✅ Research completed:")
        print(f"  - Duration: {result['research_duration']:.2f}s")
        print(f"  - Sources: {len(result['sources_used'])}")
        print(f"  - Results: {result['total_results']}")
        print(f"  - Sources used: {', '.join(result['sources_used'])}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Missing dependencies. Install with:")
        print("pip install duckduckgo-search wikipedia arxiv feedparser beautifulsoup4")
        return False
        
    except Exception as e:
        print(f"❌ Research error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

async def test_individual_sources():
    """個別の情報源をテスト"""
    print("\n=== Individual Source Tests ===")
    
    engine = FreeResearchEngine()
    
    # DuckDuckGo テスト
    try:
        ddg_results = await engine.search_duckduckgo("test query")
        print(f"✅ DuckDuckGo: {len(ddg_results)} results")
    except Exception as e:
        print(f"❌ DuckDuckGo error: {e}")
    
    # Wikipedia テスト
    try:
        wiki_results = await engine.search_wikipedia("test query")
        print(f"✅ Wikipedia: {len(wiki_results)} results")
    except Exception as e:
        print(f"❌ Wikipedia error: {e}")
    
    # ArXiv テスト
    try:
        arxiv_results = await engine.search_arxiv("machine learning")
        print(f"✅ ArXiv: {len(arxiv_results)} results")
    except Exception as e:
        print(f"❌ ArXiv error: {e}")

if __name__ == "__main__":
    # 基本テスト
    success = asyncio.run(test_research_functionality())
    
    if success:
        # 詳細テスト
        asyncio.run(test_individual_sources())
    else:
        print("\n⚠️ Please install missing dependencies and try again") 