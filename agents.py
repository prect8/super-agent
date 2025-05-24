import asyncio
import aiohttp
import base64
import io
import logging
import re
import traceback
from typing import Dict, Any, Optional, List
from PIL import Image
from config import Config

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SuperAgent:
    """統合スーパーエージェント - テキスト&マルチモーダル&リサーチ対応"""
    
    def __init__(self):
        self.config = Config()
        self.base_url = self.config.OLLAMA_BASE_URL
        
        # 会話履歴管理
        self.conversation_history = []
        self.max_history_length = 10  # 最大保持メッセージ数
        self.context_window = 5  # プロンプトに含める過去のメッセージ数
        
        # リサーチエンジンの初期化（詳細ログ付き）
        self.research_engine = None
        try:
            from research import FreeResearchEngine
            self.research_engine = FreeResearchEngine()
            logger.info("✅ Research engine initialized successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Research module not available: {e}")
        except Exception as e:
            logger.error(f"❌ Error initializing research engine: {e}")
    
    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """会話履歴にメッセージを追加"""
        message = {
            "role": role,
            "content": content,
            "timestamp": traceback.format_exc() if role == "error" else None,
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # 履歴の長さを制限
        if len(self.conversation_history) > self.max_history_length * 2:  # user + assistant のペア
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
        
        logger.info(f"Added message to history: {role} - {len(content)} chars")

    def get_conversation_context(self) -> str:
        """会話の文脈を取得"""
        if not self.conversation_history:
            return ""
        
        context_messages = self.conversation_history[-self.context_window * 2:]  # 最新のN回の対話
        context = "## 📝 会話履歴\n\n"
        
        for msg in context_messages:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context += f"{role_emoji} **{msg['role'].title()}**: {content}\n\n"
        
        context += "---\n\n"
        return context

    def extract_key_information(self, messages: List[Dict]) -> str:
        """重要な情報を抽出して要約"""
        if not messages:
            return ""
        
        key_info = []
        for msg in messages:
            content = msg.get("content", "")
            
            # 重要なキーワードを含むメッセージを抽出
            important_keywords = [
                "名前", "プロジェクト", "設定", "エラー", "問題", "解決", 
                "重要", "注意", "覚えて", "記憶", "保存", "ファイル"
            ]
            
            if any(keyword in content for keyword in important_keywords):
                summary = content[:100] + "..." if len(content) > 100 else content
                key_info.append(f"• {summary}")
        
        if key_info:
            return "## 🔑 重要な情報\n\n" + "\n".join(key_info) + "\n\n"
        return ""

    def update_model_settings(self, model: str, temperature: float):
        """モデル設定を更新（外部から呼び出し可能）"""
        logger.info(f"Model settings updated: {model}, temperature: {temperature}")
        # 必要に応じて設定を保存
        self.add_to_history("system", f"モデル設定を更新: {model}, 温度: {temperature}")

    def _analyze_task_type(self, text: str, has_image: bool = False) -> str:
        """タスクタイプを自動分析（詳細ログ付き）"""
        logger.info(f"Analyzing task type for: '{text[:50]}...' (has_image: {has_image})")
        
        text_lower = text.lower()
        
        # リサーチキーワード
        research_keywords = [
            "調べて", "研究", "情報", "最新", "ニュース", "論文", "データ", 
            "分析して", "詳しく", "検索", "リサーチ", "調査", "事実", "統計"
        ]
        
        if has_image:
            # 画像処理タスク
            for task_type in ["vision_ocr", "vision_analysis", "vision_description"]:
                keywords = self.config.TASK_KEYWORDS.get(task_type, [])
                if any(keyword in text_lower for keyword in keywords):
                    logger.info(f"Task type determined: {task_type}")
                    return task_type
            logger.info("Task type determined: vision_description (default)")
            return "vision_description"
        else:
            # リサーチ判定
            if self.research_engine and any(keyword in text_lower for keyword in research_keywords):
                logger.info("Task type determined: research")
                return "research"
            
            # テキスト処理判定
            for task_type in ["reasoning", "creative"]:
                keywords = self.config.TASK_KEYWORDS.get(task_type, [])
                if any(keyword in text_lower for keyword in keywords):
                    logger.info(f"Task type determined: {task_type}")
                    return task_type
            
            logger.info("Task type determined: general (default)")
            return "general"
    
    def _optimize_image(self, image: Image.Image) -> str:
        """画像を最適化してbase64エンコード"""
        logger.info("Optimizing image...")
        try:
            # サイズ調整
            max_size = self.config.IMAGE_MAX_SIZE
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Image resized to {new_size}")
            
            # RGB変換
            if image.mode != 'RGB':
                if image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'RGBA':
                        background.paste(image, mask=image.split()[-1])
                    else:
                        background.paste(image)
                    image = background
                else:
                    image = image.convert('RGB')
                logger.info("Image converted to RGB")
            
            # Base64エンコード
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.config.IMAGE_QUALITY, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logger.info(f"Image encoded (base64 length: {len(encoded)})")
            return encoded
            
        except Exception as e:
            logger.error(f"Image optimization error: {e}")
            raise
    
    def _create_prompt(self, query: str, task_type: str, research_data: Optional[Dict] = None) -> str:
        """タスクタイプに応じたプロンプトを作成（会話履歴付き）"""
        logger.info(f"Creating prompt for task type: {task_type}")
        
        # 会話履歴を含める
        conversation_context = self.get_conversation_context()
        key_information = self.extract_key_information(self.conversation_history)
        
        base_prompt = f"{conversation_context}{key_information}**現在の質問**: {query}\n\n"
        
        # リサーチタスクの場合
        if task_type == "research" and research_data:
            research_summary = self._generate_research_summary(research_data)
            logger.info("Created research-based prompt")
            return f"""
{conversation_context}{key_information}

あなたは情報分析の専門家です。会話履歴を参考にしながら、以下のリサーチ結果を基に、ユーザーの質問に詳細に回答してください。

ユーザーの質問: {query}

リサーチ結果:
{research_summary}

以下の構造で回答してください：
1. **要約**: 重要なポイントの概要
2. **詳細分析**: 各情報源からの知見
3. **会話の文脈**: 過去の対話との関連性
4. **信頼性評価**: 情報の信頼度と根拠
5. **追加の考察**: 専門的な分析と見解
6. **関連情報**: さらに調べるべき点

客観的で分析的な回答を心がけ、会話の流れを意識して情報源を明示してください。
"""
        
        # その他のタスクタイプ
        prompts = {
            "reasoning": base_prompt + """
あなたは論理的思考の専門家です。会話履歴を参考にしながら、以下の手順で詳細に分析してください：
1. 過去の対話との関連性の確認
2. 問題の本質的な理解
3. 関連要因の分析  
4. 段階的な推論過程
5. 論理的な結論
思考過程を明確に示し、会話の流れを意識してください。
""",
            "creative": base_prompt + """
あなたは創造性豊かな専門家です。会話履歴のテーマや雰囲気を活かしながら、以下の要素を含めて創作してください：
1. 過去の対話で言及されたテーマの活用
2. 独創的なアイデア
3. 鮮明で美しい描写
4. 感情に訴える表現
5. 実用的な提案
自由で魅力的な発想を展開し、会話の継続性を保ってください。
""",
            "general": base_prompt + """
あなたは幅広い知識を持つ専門家です。会話履歴を踏まえて、以下を含めて回答してください：
1. 過去の対話との関連性
2. 正確で詳細な情報
3. 具体的な例
4. 実用的なアドバイス
5. 次のステップ
分かりやすく実用的に説明し、会話の流れを意識してください。
""",
            "vision_description": base_prompt + """
画像を詳しく説明してください。また、過去の対話で言及された内容との関連があれば指摘してください：
1. 全体的な構図と主要素
2. 色彩、雰囲気、質感
3. 人物、物体、風景の詳細
4. 場所や時間の推測
5. 印象的な特徴
6. 会話履歴との関連性
""",
            "vision_analysis": base_prompt + """
画像を専門的に分析してください。過去の対話内容も考慮に入れてください：
1. 技術的特徴と手法
2. 構成要素の詳細分析
3. 背景情報と文脈
4. 特筆すべき要素
5. 専門的な考察
6. 会話履歴との関連分析
""",
            "vision_ocr": base_prompt + """
画像内のテキストを正確に読み取ってください。過去の対話で関連する内容があれば言及してください：
1. 文字の正確な文字起こし
2. レイアウト構造の保持
3. 読み取れない部分の明記
4. 言語の適切な認識
5. 会話履歴との関連性
"""
        }
        
        prompt = prompts.get(task_type, prompts["general"])
        logger.info(f"Created prompt with context (length: {len(prompt)})")
        return prompt
    
    def _generate_research_summary(self, research_data: Dict[str, Any]) -> str:
        """リサーチ結果のサマリーを生成"""
        logger.info("Generating research summary...")
        if not research_data:
            logger.warning("No research data available")
            return ""
        
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
        
        logger.info(f"Generated summary (length: {len(summary)})")
        return summary
    
    def _select_model(self, task_type: str) -> str:
        """タスクタイプに応じてモデルを選択"""
        logger.info(f"Selecting model for task type: {task_type}")
        model_mapping = {
            "research": self.config.MODELS["reasoning"],  # 分析力の高いモデル
            "reasoning": self.config.MODELS["reasoning"],
            "creative": self.config.MODELS["creative"], 
            "general": self.config.MODELS["general"],
            "vision_description": self.config.MODELS["vision"],
            "vision_analysis": self.config.MODELS.get("vision_detailed", self.config.MODELS["vision"]),
            "vision_ocr": self.config.MODELS.get("vision_fast", self.config.MODELS["vision"])
        }
        model = model_mapping.get(task_type, self.config.MODELS["general"])
        logger.info(f"Selected model: {model}")
        return model
    
    def _get_temperature(self, task_type: str) -> float:
        """タスクタイプに応じて温度を設定"""
        logger.info(f"Setting temperature for task type: {task_type}")
        temperatures = {
            "research": 0.3,
            "reasoning": 0.1,
            "general": 0.7,
            "creative": 0.9,
            "vision_description": 0.7,
            "vision_analysis": 0.3,
            "vision_ocr": 0.1
        }
        temp = temperatures.get(task_type, 0.7)
        logger.info(f"Temperature set to: {temp}")
        return temp
    
    async def _call_api(self, model: str, prompt: str, image_base64: Optional[str] = None, temperature: float = 0.7) -> Dict[str, Any]:
        """Ollama API呼び出し"""
        url = f"{self.base_url}/api/chat"
        logger.info(f"Calling Ollama API: {url}")
        
        message = {"role": "user", "content": prompt}
        if image_base64:
            message["images"] = [image_base64]
            logger.info("Image included in API call")
        
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 2048
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=self.config.OLLAMA_TIMEOUT) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("API call successful")
                        return {
                            "success": True,
                            "response": result.get("message", {}).get("content", "応答を取得できませんでした"),
                            "model_info": {
                                "model": result.get("model", model),
                                "eval_count": result.get("eval_count", 0),
                                "total_duration": result.get("total_duration", 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        return {
                            "success": False,
                            "response": f"APIエラー: {error_text}",
                            "error": f"Status {response.status}"
                        }
        except Exception as e:
            logger.error(f"API call error: {e}")
            return {
                "success": False,
                "response": f"エラーが発生しました: {str(e)}",
                "error": str(e)
            }
    
    async def process_query(self, text: str, image: Optional[Image.Image] = None, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """メインの処理関数（会話履歴対応）"""
        logger.info(f"Processing query: '{text[:100]}...'")
        
        # 外部から会話履歴を受け取る場合
        if conversation_history:
            self.conversation_history = conversation_history[-self.max_history_length * 2:]
        
        # ユーザーメッセージを履歴に追加
        self.add_to_history("user", text, {"has_image": image is not None})
        
        try:
            # タスク分析
            task_type = self._analyze_task_type(text, image is not None)
            logger.info(f"Task type: {task_type}")
            
            research_data = None
            
            # リサーチ実行
            if task_type == "research" and self.research_engine:
                logger.info("Starting deep research...")
                try:
                    research_data = await self.research_engine.conduct_deep_research(text)
                    logger.info(f"Research completed: {research_data['total_results']} results")
                except Exception as e:
                    logger.error(f"Research failed: {e}")
                    task_type = "reasoning"  # フォールバック
                    logger.info(f"Fallback to task type: {task_type}")
            
            # モデル選択
            model = self._select_model(task_type)
            temperature = self._get_temperature(task_type)
            logger.info(f"Selected model: {model}, temperature: {temperature}")
            
            # プロンプト作成
            prompt = self._create_prompt(text, task_type, research_data)
            logger.info(f"Prompt created (length: {len(prompt)})")
            
            # 画像処理
            image_base64 = None
            if image:
                image_base64 = self._optimize_image(image)
                logger.info(f"Image optimized (base64 length: {len(image_base64)})")
            
            # API呼び出し
            logger.info("Calling Ollama API...")
            result = await self._call_api(model, prompt, image_base64, temperature)
            logger.info(f"API call completed: success={result['success']}")
            
            # アシスタントの応答を履歴に追加
            if result["success"]:
                self.add_to_history("assistant", result["response"], {
                    "task_type": task_type,
                    "model": model,
                    "temperature": temperature
                })
            
            return {
                "task_type": task_type,
                "response": result["response"],
                "success": result["success"],
                "has_image": image is not None,
                "research_data": research_data,
                "conversation_history": self.conversation_history,
                "model_info": {
                    "model": model,
                    "temperature": temperature,
                    **result.get("model_info", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_query: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # エラーを履歴に追加
            self.add_to_history("error", f"処理エラー: {str(e)}")
            
            return {
                "task_type": "error",
                "response": f"処理エラー: {str(e)}",
                "success": False,
                "has_image": image is not None,
                "conversation_history": self.conversation_history,
                "model_info": {"model": "error", "temperature": 0}
            } 