import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import aiohttp
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

class GeneralAgent:
    """一般的なタスクと要約を行うエージェント"""
    
    def __init__(self):
        self.model = "deepseek-r1:14b"
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger("GeneralAgent")
        logger.setLevel(logging.INFO)
        
        # ファイルハンドラの設定
        fh = logging.FileHandler("agent_interaction.log")
        fh.setLevel(logging.INFO)
        
        # フォーマッタの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデルの設定情報を取得"""
        return {
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }

    async def process(self, query: str) -> str:
        """クエリを処理し、要約と分析を生成"""
        try:
            self.logger.info(f"処理開始: {query}")
            
            # プロンプトの構築
            prompt = self._build_processing_prompt(query)
            
            # APIリクエストの送信
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/completion",
                    headers={
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "prompt": f"あなたは一般的なタスク処理と要約の専門家です。\n\n{prompt}",
                        "stream": False,
                        "temperature": 0.7,
                        "max_tokens": 2000
                    }
                ) as response:
                    if response.status != 200:
                        raise Exception(f"API error: {response.status}")
                    
                    result = await response.json()
                    return result["response"]
            
        except Exception as e:
            self.logger.error(f"処理中にエラーが発生: {str(e)}")
            raise
    
    def _build_processing_prompt(self, query: str) -> str:
        """処理用のプロンプトを構築"""
        return f"""
以下の内容を処理し、要約と分析を提供してください：

{query}

以下の形式で回答してください：
1. 主要なポイントの要約
2. 詳細な分析
3. 実用的な提案
4. 結論
"""
    
    def _structure_response(self, response: str) -> Dict[str, Any]:
        """応答を構造化"""
        return {
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "model": self.model,
            "confidence_score": 0.9  # 仮の値
        }
    
    async def validate_response(self, response: Dict[str, Any]) -> bool:
        """応答の妥当性を検証"""
        try:
            # 基本的な検証ロジック
            required_fields = ["response", "model", "confidence_score"]
            return all(field in response for field in required_fields)
        except Exception as e:
            self.logger.error(f"検証中にエラーが発生: {str(e)}")
            return False 