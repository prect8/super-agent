from typing import Dict, List, Any, Optional
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
import re
import requests
import json
import aiohttp
from datetime import datetime

from agents.reasoning_agent import ReasoningAgent
from agents.general_agent import GeneralAgent
from agents.creative_agent import CreativeAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ファイルハンドラの設定
fh = logging.FileHandler("agent_interaction.log")
fh.setLevel(logging.INFO)

# フォーマッタの設定
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class TaskType(Enum):
    """タスクの種類を定義する列挙型"""
    REASONING = "reasoning"  # 論理的推論・分析
    GENERAL = "general"      # 汎用的なタスク
    CREATIVE = "creative"    # 創作・アイデア生成
    COMPLEX = "complex"      # 複合的なタスク

@dataclass
class TaskAnalysis:
    """タスク分析結果を保持するデータクラス"""
    task_type: TaskType
    confidence: float
    required_agents: List[str]
    reasoning: str

class MasterRouter:
    """タスクを分析し、適切なエージェントに振り分けるルーター"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reasoning_agent = ReasoningAgent()
        self.general_agent = GeneralAgent()
        self.creative_agent = CreativeAgent()
        
        # エージェントの辞書を作成
        self.agents = {
            "reasoning": self.reasoning_agent,
            "general": self.general_agent,
            "creative": self.creative_agent
        }
        
        # タスクタイプ判定のためのキーワード
        self.task_keywords = {
            TaskType.REASONING: [
                "分析", "推論", "論理", "因果", "なぜ", "理由", "根拠",
                "比較", "評価", "検証", "仮説", "証明", "考察"
            ],
            TaskType.GENERAL: [
                "説明", "解説", "要約", "整理", "分類", "定義",
                "方法", "手順", "使い方", "設定", "技術", "仕組み"
            ],
            TaskType.CREATIVE: [
                "創作", "アイデア", "ストーリー", "文章", "表現",
                "比喩", "例え", "描写", "物語", "詩", "歌詞"
            ]
        }
    
    def _analyze_task_type(self, query: str) -> TaskType:
        """
        クエリを分析して適切なタスクタイプを判定
        
        Args:
            query (str): ユーザーのクエリ
            
        Returns:
            TaskType: 判定されたタスクタイプ
        """
        # キーワードの出現回数をカウント
        type_scores = {task_type: 0 for task_type in TaskType}
        
        for task_type, keywords in self.task_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    type_scores[task_type] += 1
        
        # 最も関連性の高いタスクタイプを特定
        max_score = max(type_scores.values())
        if max_score == 0:
            return TaskType.GENERAL  # デフォルトは汎用タスク
        
        # 複数のタスクタイプが同点の場合は複合タスクとして扱う
        if list(type_scores.values()).count(max_score) > 1:
            return TaskType.COMPLEX
            
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    async def _coordinate_agents(self, query: str, task_type: TaskType) -> str:
        """
        タスクタイプに応じて適切なエージェントを選択し、タスクを実行
        
        Args:
            query (str): ユーザーのクエリ
            task_type (TaskType): 判定されたタスクタイプ
            
        Returns:
            str: エージェントからの応答
        """
        try:
            if task_type == TaskType.COMPLEX:
                # 複合タスクの場合は複数のエージェントで協力
                tasks = []
                
                # 推論エージェントで問題を分析
                analysis_task = self.reasoning_agent.analyze(
                    f"以下の問題を分析し、必要な処理ステップを特定してください：\n{query}"
                )
                tasks.append(analysis_task)
                
                # 分析結果を待機
                analysis_result = await asyncio.gather(*tasks)
                
                # 分析結果に基づいて適切なエージェントを選択
                if any(keyword in analysis_result[0].lower() for keyword in self.task_keywords[TaskType.CREATIVE]):
                    return await self.creative_agent.create(query)
                elif any(keyword in analysis_result[0].lower() for keyword in self.task_keywords[TaskType.REASONING]):
                    return await self.reasoning_agent.analyze(query)
                else:
                    return await self.general_agent.process(query)
            
            elif task_type == TaskType.REASONING:
                return await self.reasoning_agent.analyze(query)
            elif task_type == TaskType.GENERAL:
                return await self.general_agent.process(query)
            elif task_type == TaskType.CREATIVE:
                return await self.creative_agent.create(query)
            
        except Exception as e:
            self.logger.error(f"Error in agent coordination: {str(e)}")
            return f"タスク処理中にエラーが発生しました: {str(e)}"
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        ユーザーのクエリを処理し、適切なエージェントに振り分ける
        
        Args:
            query (str): ユーザーのクエリ
            
        Returns:
            Dict[str, Any]: 処理結果とメタデータ
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # タスクタイプを分析
            task_type = self._analyze_task_type(query)
            self.logger.info(f"Detected task type: {task_type.value}")
            
            # 適切なエージェントを選択してタスクを実行
            response = await self._coordinate_agents(query, task_type)
            
            return {
                "success": True,
                "task_type": task_type.value,
                "response": response,
                "model_info": self._get_agent_model_info(task_type)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_agent_model_info(self, task_type: TaskType) -> Dict[str, Any]:
        """
        タスクタイプに応じたエージェントのモデル情報を取得
        
        Args:
            task_type (TaskType): タスクタイプ
            
        Returns:
            Dict[str, Any]: モデル情報
        """
        if task_type == TaskType.REASONING:
            return self.reasoning_agent.get_model_info()
        elif task_type == TaskType.GENERAL:
            return self.general_agent.get_model_info()
        elif task_type == TaskType.CREATIVE:
            return self.creative_agent.get_model_info()
        else:
            return {
                "models": [
                    self.reasoning_agent.get_model_info(),
                    self.general_agent.get_model_info(),
                    self.creative_agent.get_model_info()
                ]
            }

class SuperAgentTeam:
    """複数のエージェントを統括するスーパーチーム"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.models = {
            "reasoning": "deepseek-r1:14b",
            "general": "deepseek-r1:14b", 
            "creative": "gemma3:latest"
        }
        
    def _analyze_task_type(self, query: str) -> str:
        """タスクタイプを分析"""
        reasoning_keywords = ['分析', '推論', '論理', '証明', '問題解決', '比較', '検討', '評価', 'なぜ', '理由']
        creative_keywords = ['創作', '小説', 'アイデア', 'ブレスト', '企画', 'プロット', '物語', '詩', '作って']
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in reasoning_keywords):
            return "reasoning"
        elif any(keyword in query_lower for keyword in creative_keywords):
            return "creative"
        else:
            return "general"
    
    def _process_thinking_response(self, response: str) -> Dict[str, str]:
        """<think></think>タグを処理して思考過程と最終回答を分離"""
        try:
            # <think>...</think>タグを抽出
            thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            
            if thinking_match:
                thinking_process = thinking_match.group(1).strip()
                # <think></think>タグを除いた部分を最終回答とする
                final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            else:
                thinking_process = ""
                final_answer = response
            
            return {
                "thinking": thinking_process,
                "answer": final_answer
            }
        except Exception as e:
            logger.error(f"Error processing thinking response: {str(e)}")
            return {
                "thinking": "",
                "answer": response
            }
    
    async def _call_ollama_chat_api(self, model: str, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Ollama Chat APIを呼び出し"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 2048  # 長い回答を許可
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=120) as response:
                    if response.status == 200:
                        result = await response.json()
                        raw_response = result.get("message", {}).get("content", "応答を取得できませんでした。")
                        
                        # 推論過程の処理
                        processed = self._process_thinking_response(raw_response)
                        
                        return {
                            "success": True,
                            "thinking": processed["thinking"],
                            "answer": processed["answer"],
                            "raw_response": raw_response,
                            "model_info": {
                                "model": result.get("model", model),
                                "eval_count": result.get("eval_count", 0),
                                "eval_duration": result.get("eval_duration", 0),
                                "total_duration": result.get("total_duration", 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"API call failed with status {response.status}: {error_text}")
                        return {
                            "success": False,
                            "error": f"APIエラー: ステータスコード {response.status}",
                            "thinking": "",
                            "answer": f"APIエラーが発生しました: {error_text}"
                        }
                        
        except asyncio.TimeoutError:
            logger.error("API call timed out")
            return {
                "success": False,
                "error": "タイムアウト",
                "thinking": "",
                "answer": "タイムアウトエラー: 応答に時間がかかりすぎました。"
            }
        except Exception as e:
            logger.error(f"Error calling Ollama Chat API: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "thinking": "",
                "answer": f"エラーが発生しました: {str(e)}"
            }
    
    def _create_specialized_prompt(self, query: str, task_type: str) -> str:
        """タスクタイプに応じた専門的なプロンプトを作成"""
        
        if task_type == "reasoning":
            return f"""あなたは論理的思考と深い推論を得意とする専門家です。
以下の質問について、段階的に思考過程を示しながら詳細に分析してください。

質問: {query}

以下の構造で回答してください：
1. 問題の本質的な理解
2. 関連する要因や背景の分析
3. 複数の視点からの検討
4. 論理的な推論過程
5. 根拠に基づいた結論

思考過程は<think>タグで囲んで示し、最終的な結論も分かりやすく提示してください。
例：
<think>
1. まず、問題の本質を理解するために...
2. 次に、関連する要因を分析すると...
3. 複数の視点から検討すると...
</think>

最終的な結論は以下の通りです..."""

        elif task_type == "creative":
            return f"""あなたは創造性豊かで想像力に溢れるクリエイティブ専門家です。
以下のリクエストについて、独創的で魅力的なアイデアを提供してください。

リクエスト: {query}

以下の要素を含めて創造的に回答してください：
1. 独創的で斬新なアイデア
2. 具体的で鮮明な描写
3. 感情に訴える表現
4. 実用性も考慮した提案
5. 発展可能性のあるコンセプト

自由な発想で、魅力的で実用的なアイデアを提示してください。"""

        else:  # general
            return f"""あなたは幅広い知識と経験を持つ総合的な専門家です。
以下の質問について、正確で実用的な情報を分かりやすく説明してください。

質問: {query}

以下の点を含めて包括的に回答してください：
1. 質問の核心的な内容の説明
2. 正確で詳細な情報や知識
3. 具体例やケーススタディ
4. 実践的なアドバイスや手順
5. 追加の学習リソースや次のステップ

初心者にも理解しやすく、実用的な情報を提供してください。"""
    
    async def handle_query(self, query: str) -> Dict[str, Any]:
        """クエリを処理してレスポンスを返す"""
        try:
            # タスクタイプの分析
            task_type = self._analyze_task_type(query)
            selected_model = self.models[task_type]
            
            # 専門プロンプトの作成
            specialized_prompt = self._create_specialized_prompt(query, task_type)
            
            # 温度設定（推論タスクは低め、創作タスクは高め）
            temperature = 0.1 if task_type == "reasoning" else 0.9 if task_type == "creative" else 0.7
            
            # APIコール
            result = await self._call_ollama_chat_api(selected_model, specialized_prompt, temperature)
            
            if result["success"]:
                return {
                    "task_type": task_type,
                    "response": result["answer"],
                    "thinking_process": result["thinking"],
                    "raw_response": result["raw_response"],
                    "model_info": {
                        "model": selected_model,
                        "temperature": temperature,
                        "timeout": 120,
                        **result.get("model_info", {})
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "task_type": "error",
                    "response": result["answer"],
                    "thinking_process": "",
                    "error": result["error"],
                    "model_info": {
                        "model": selected_model,
                        "temperature": temperature,
                        "timeout": 120
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in handle_query: {str(e)}")
            return {
                "task_type": "error",
                "response": f"処理中にエラーが発生しました: {str(e)}",
                "thinking_process": "",
                "model_info": {
                    "model": "error",
                    "temperature": 0,
                    "timeout": 0
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def get_team_info(self) -> Dict[str, Any]:
        """チームの情報を取得"""
        return {
            "team_name": "SuperAgentTeam",
            "agents": {
                name: agent.get_model_info()
                for name, agent in self.router.agents.items()
            },
            "capabilities": [
                "複雑な推論と分析",
                "汎用的なタスク処理",
                "創作活動とアイデア生成",
                "複合タスクの統合処理"
            ]
        } 