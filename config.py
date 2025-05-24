import os
from typing import Dict, List

class Config:
    """アプリケーション設定"""
    
    # Ollama設定
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_TIMEOUT = 180
    
    # 利用可能なモデル設定
    MODELS = {
        # テキスト専用モデル
        "reasoning": "deepseek-r1:14b",      # 推論・分析
        "general": "deepseek-r1:14b",        # 汎用
        "creative": "gemma3:latest",         # 創作
        
        # マルチモーダルモデル
        "vision": "llava:latest",            # 画像+テキスト
        "vision_detailed": "llava:13b",      # 詳細画像分析
        "vision_fast": "moondream:latest"    # 高速画像処理
    }
    
    # タスク分類キーワード（agents.pyで参照されるキー名と一致させる）
    TASK_KEYWORDS = {
        "reasoning": ["分析", "推論", "論理", "証明", "問題解決", "比較", "検討", "評価", "なぜ", "理由", "考察", "検証"],
        "creative": ["創作", "小説", "アイデア", "ブレスト", "企画", "プロット", "物語", "詩", "作って", "創造", "発想"],
        "vision_ocr": ["読んで", "文字", "テキスト", "書いて", "文章", "読み取", "ocr", "文字起こし"],
        "vision_analysis": ["分析", "詳しく", "詳細", "調べて", "解析", "検証", "専門的"],
        "vision_description": ["説明", "描写", "何が", "なに", "見える", "写っ", "どんな"]
    }
    
    # UI設定
    PAGE_CONFIG = {
        "page_title": "🤖 スーパーエージェント",
        "page_icon": "🤖",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # 画像最適化設定
    IMAGE_MAX_SIZE = 1024
    IMAGE_QUALITY = 85
    
    # アプリケーション設定
    APP_SETTINGS = {
        "title": "🤖 スーパーエージェント",
        "description": "複数の専門エージェントが協調して質問に答えます",
        "theme": "light",
        "max_history": 100
    }
    
    # ログ設定
    LOG_SETTINGS = {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "agent_interaction.log"
    } 