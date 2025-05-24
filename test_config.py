from config import Config
from agents import SuperAgent

def test_config_integrity():
    """設定の整合性をテスト"""
    config = Config()
    
    print("=== Config Test ===")
    
    # MODELS辞書の確認
    print("Models:")
    for key, model in config.MODELS.items():
        print(f"  {key}: {model}")
    
    # TASK_KEYWORDS辞書の確認
    print("\nTask Keywords:")
    for task_type, keywords in config.TASK_KEYWORDS.items():
        print(f"  {task_type}: {keywords[:3]}...")
    
    # SuperAgentの初期化テスト
    print("\n=== SuperAgent Test ===")
    try:
        agent = SuperAgent()
        print("✅ SuperAgent initialization successful")
        
        # タスク分析テスト
        test_queries = [
            ("AIについて分析して", False),
            ("小説を書いて", False),
            ("これは何ですか？", True)
        ]
        
        for query, has_image in test_queries:
            task_type = agent._analyze_task_type(query, has_image)
            print(f"  Query: '{query}' -> Task: {task_type}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_error_conditions():
    """エラー条件のテスト"""
    print("\n=== Error Condition Tests ===")
    
    config = Config()
    agent = SuperAgent()
    
    # リサーチエンジンの状態確認
    print(f"Research engine available: {agent.research_engine is not None}")
    
    # さまざまなクエリでのタスク分析テスト
    test_queries = [
        "これを分析してください",  # reasoning候補
        "最新の情報を調べて",      # research候補 
        "詩を作って",              # creative
        "何か教えて"               # general
    ]
    
    for query in test_queries:
        try:
            task_type = agent._analyze_task_type(query, False)
            model = agent._select_model(task_type)
            temp = agent._get_temperature(task_type)
            print(f"  '{query}' -> {task_type} (model: {model}, temp: {temp})")
        except Exception as e:
            print(f"  ❌ Error with '{query}': {e}")
    
    # エッジケースのテスト
    print("\nEdge Cases:")
    edge_cases = [
        "",  # 空文字列
        "   ",  # 空白のみ
        "!@#$%^&*()",  # 特殊文字
        "a" * 1000,  # 長い文字列
    ]
    
    for query in edge_cases:
        try:
            task_type = agent._analyze_task_type(query, False)
            print(f"  Edge case '{query[:20]}...' -> {task_type}")
        except Exception as e:
            print(f"  ❌ Error with edge case: {e}")
    
    # 画像処理のエラーケース
    print("\nImage Processing Edge Cases:")
    try:
        # 存在しない画像ファイル
        from PIL import Image
        image = Image.open("nonexistent.jpg")
        task_type = agent._analyze_task_type("これは何ですか？", True)
        print(f"  Image processing -> {task_type}")
    except Exception as e:
        print(f"  ❌ Image processing error: {e}")

if __name__ == "__main__":
    test_config_integrity()
    test_error_conditions() 