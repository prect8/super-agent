import asyncio
import aiohttp
import base64
from typing import Dict, Any, List, Optional, Union
import logging
from PIL import Image
import io
import re

logger = logging.getLogger(__name__)

class MultimodalAgentTeam:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        
        # マルチモーダルモデルの定義
        self.multimodal_models = {
            "vision_general": "llava:latest",      # 汎用画像処理
            "vision_detailed": "llava:13b",        # 詳細分析
            "vision_fast": "moondream:latest",     # 高速処理
            "vision_compact": "bakllava:latest"    # 軽量版
        }
        
        # テキストオンリーモデル（フォールバック用）
        self.text_models = {
            "reasoning": "deepseek-r1:14b",
            "general": "deepseek-r1:14b", 
            "creative": "gemma3:latest"
        }
        
    def _optimize_image(self, image: Image.Image, max_size: int = 512) -> Image.Image:
        """画像のサイズとフォーマットを最適化"""
        # アスペクト比を保持してリサイズ
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # RGBに変換（RGBA、Grayscaleなどを統一）
        if image.mode != 'RGB':
            # 透明度がある場合は白背景に合成
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        return image
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """画像をbase64エンコード"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _analyze_multimodal_task_type(self, text_query: str, has_image: bool) -> str:
        """マルチモーダルタスクのタイプを分析"""
        if not has_image:
            # 画像がない場合は従来のテキスト分析
            reasoning_keywords = ['分析', '推論', '論理', '証明', '問題解決', '比較']
            creative_keywords = ['創作', '小説', 'アイデア', 'ブレスト', '企画']
            
            query_lower = text_query.lower()
            if any(keyword in query_lower for keyword in reasoning_keywords):
                return "text_reasoning"
            elif any(keyword in query_lower for keyword in creative_keywords):
                return "text_creative"
            else:
                return "text_general"
        
        # 画像がある場合のタスク分析
        query_lower = text_query.lower()
        
        ocr_keywords = ['読んで', '文字', 'テキスト', '書いて', '文章', '読み取', 'ocr']
        analysis_keywords = ['分析', '詳しく', '詳細', '調べて', '解析', '検証']
        description_keywords = ['説明', '描写', '何が', 'なに', '見える', '写っ']
        creative_keywords = ['物語', 'ストーリー', '創作', '想像', 'アイデア']
        
        if any(keyword in query_lower for keyword in ocr_keywords):
            return "vision_ocr"
        elif any(keyword in query_lower for keyword in analysis_keywords):
            return "vision_analysis"
        elif any(keyword in query_lower for keyword in creative_keywords):
            return "vision_creative"
        else:
            return "vision_description"
    
    def _create_multimodal_prompt(self, query: str, task_type: str) -> str:
        """タスクタイプに応じたマルチモーダルプロンプトを作成"""
        base_prompt = f"ユーザーの質問: {query}\n\n"
        
        if task_type == "vision_ocr":
            return base_prompt + """
画像に含まれるテキストを正確に読み取って文字起こししてください。
以下の点に注意してください：
1. 文字の正確な読み取り
2. レイアウトや構造の保持
3. 読み取れない部分は明記
4. 日本語、英語などの言語を適切に認識
"""
        
        elif task_type == "vision_analysis":
            return base_prompt + """
画像を詳細に分析して包括的な情報を提供してください。
以下の観点から分析してください：
1. 主要な要素と構成
2. 色彩、質感、スタイル
3. 技術的な特徴や手法
4. 背景や文脈の情報
5. 特筆すべき詳細や特徴
"""
        
        elif task_type == "vision_creative":
            return base_prompt + """
画像からインスピレーションを得て、創造的なコンテンツを作成してください。
以下の要素を含めて創作してください：
1. 画像から感じられる雰囲気や感情
2. 物語性のある表現
3. 想像力豊かな解釈
4. 詩的で美しい描写
5. ユニークな視点や発想
"""
        
        else:  # vision_description
            return base_prompt + """
画像を分かりやすく説明してください。
以下の点を含めて説明してください：
1. 全体的な構図と主要な被写体
2. 色彩、明度、雰囲気
3. 人物、物体、風景の詳細
4. 場所や時間の推測
5. 印象的な特徴や注目点
"""
    
    async def _call_multimodal_api(self, model: str, prompt: str, image_base64: str, temperature: float = 0.7) -> Dict[str, Any]:
        """マルチモーダルAPI呼び出し"""
        url = f"{self.base_url}/api/chat"
        
        # メッセージの構築
        message = {
            "role": "user",
            "content": prompt
        }
        
        if image_base64:
            message["images"] = [image_base64]
        
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 3072  # 長い回答を許可
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=180) as response:  # タイムアウトを長めに
                    if response.status == 200:
                        result = await response.json()
                        raw_response = result.get("message", {}).get("content", "応答を取得できませんでした。")
                        
                        return {
                            "success": True,
                            "response": raw_response,
                            "model_info": {
                                "model": result.get("model", model),
                                "eval_count": result.get("eval_count", 0),
                                "total_duration": result.get("total_duration", 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Multimodal API call failed: {response.status} - {error_text}")
                        return {
                            "success": False,
                            "error": f"APIエラー: {response.status}",
                            "response": f"APIエラーが発生しました: {error_text}"
                        }
                        
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "タイムアウト",
                "response": "処理がタイムアウトしました。画像が大きすぎる可能性があります。"
            }
        except Exception as e:
            logger.error(f"Error in multimodal API call: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"エラーが発生しました: {str(e)}"
            }
    
    async def handle_multimodal_query(self, text_query: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """マルチモーダルクエリの処理"""
        try:
            # タスクタイプの分析
            task_type = self._analyze_multimodal_task_type(text_query, image is not None)
            
            if image is not None:
                # 画像処理
                optimized_image = self._optimize_image(image)
                image_base64 = self._encode_image_to_base64(optimized_image)
                
                # マルチモーダルモデルの選択
                if task_type == "vision_analysis":
                    model = self.multimodal_models.get("vision_detailed", "llava:latest")
                    temperature = 0.3
                elif task_type == "vision_creative":
                    model = self.multimodal_models.get("vision_general", "llava:latest")
                    temperature = 0.8
                else:
                    model = self.multimodal_models.get("vision_general", "llava:latest")
                    temperature = 0.7
                
                # プロンプトの作成
                specialized_prompt = self._create_multimodal_prompt(text_query, task_type)
                
                # API呼び出し
                result = await self._call_multimodal_api(model, specialized_prompt, image_base64, temperature)
                
            else:
                # テキストのみの処理（既存のロジック使用）
                from .super_agent_team import SuperAgentTeam
                text_team = SuperAgentTeam()
                result = await text_team.handle_query(text_query)
                return result
            
            if result["success"]:
                return {
                    "task_type": task_type,
                    "response": result["response"],
                    "has_image": image is not None,
                    "image_processed": image is not None,
                    "model_info": {
                        "model": model,
                        "temperature": temperature,
                        "timeout": 180,
                        **result.get("model_info", {})
                    }
                }
            else:
                return {
                    "task_type": "error",
                    "response": result["response"],
                    "error": result["error"],
                    "has_image": image is not None,
                    "model_info": {
                        "model": model if 'model' in locals() else "error",
                        "temperature": 0,
                        "timeout": 180
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in handle_multimodal_query: {str(e)}")
            return {
                "task_type": "error",
                "response": f"処理中にエラーが発生しました: {str(e)}",
                "has_image": image is not None,
                "model_info": {
                    "model": "error",
                    "temperature": 0,
                    "timeout": 0
                }
            } 