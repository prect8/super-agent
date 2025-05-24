import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional, Callable
from functools import lru_cache, wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクスを保持するデータクラス"""
    response_time: float
    memory_usage: float
    cpu_usage: float
    model_switch_time: float
    cache_hits: int
    cache_misses: int
    timestamp: datetime

class PerformanceMonitor:
    """パフォーマンス監視と最適化を担当するクラス"""
    
    def __init__(self):
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=100)
        self.process = psutil.Process()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # モデル固有の最適化パラメータ
        self.model_params = {
            "deepseek-r1:14b": {
                "reasoning": {
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "batch_size": 4
                },
                "general": {
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "batch_size": 8
                }
            },
            "gemma3:latest": {
                "creative": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "stream_chunk_size": 64
                }
            }
        }
    
    def measure_performance(self, func: Callable) -> Callable:
        """パフォーマンス計測用デコレータ"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = psutil.cpu_percent()
            
            try:
                result = await func(*args, **kwargs)
                
                # メトリクスの記録
                metrics = PerformanceMetrics(
                    response_time=time.time() - start_time,
                    memory_usage=self.process.memory_info().rss / 1024 / 1024 - start_memory,
                    cpu_usage=psutil.cpu_percent() - start_cpu,
                    model_switch_time=0.0,  # モデル切り替え時間は別途計測
                    cache_hits=self.cache_hits,
                    cache_misses=self.cache_misses,
                    timestamp=datetime.now()
                )
                self.metrics_history.append(metrics)
                
                return result
                
            except Exception as e:
                logger.error(f"Performance measurement error: {str(e)}")
                raise
        
        return wrapper
    
    @lru_cache(maxsize=100)
    def get_optimized_params(self, model: str, task_type: str) -> Dict[str, Any]:
        """タスクタイプに応じた最適化パラメータを取得"""
        return self.model_params.get(model, {}).get(task_type, {})
    
    def optimize_model_switch(self, current_model: str, target_model: str) -> float:
        """モデル切り替えの最適化"""
        start_time = time.time()
        
        # モデル切り替えのオーバーヘッドを最小化
        if current_model == target_model:
            return 0.0
        
        # メモリの解放
        self._cleanup_memory()
        
        # モデル固有の最適化
        if target_model == "deepseek-r1:14b":
            self._optimize_deepseek()
        elif target_model == "gemma3:latest":
            self._optimize_gemma()
        
        return time.time() - start_time
    
    def _cleanup_memory(self):
        """メモリ使用量の最適化"""
        import gc
        gc.collect()
        
        # キャッシュのクリーンアップ
        if len(self.cache) > 1000:
            self.cache.clear()
    
    def _optimize_deepseek(self):
        """deepseek-r1:14bの最適化"""
        # 推論タスクの最適化
        self.model_params["deepseek-r1:14b"]["reasoning"].update({
            "temperature": 0.1,
            "max_tokens": 4096,
            "batch_size": 4
        })
        
        # コンテキスト長の最適化
        self.model_params["deepseek-r1:14b"]["general"].update({
            "temperature": 0.3,
            "max_tokens": 2048,
            "batch_size": 8
        })
    
    def _optimize_gemma(self):
        """gemma3:latestの最適化"""
        # 創作タスクの最適化
        self.model_params["gemma3:latest"]["creative"].update({
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream_chunk_size": 64
        })
    
    async def process_batch(self, tasks: List[Dict[str, Any]], model: str) -> List[Any]:
        """バッチ処理の最適化"""
        batch_size = self.get_optimized_params(model, "general")["batch_size"]
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self._process_single_task(task, model)
                for task in batch
            ])
            results.extend(batch_results)
        
        return results
    
    async def _process_single_task(self, task: Dict[str, Any], model: str) -> Any:
        """単一タスクの処理"""
        # キャッシュチェック
        cache_key = f"{model}:{task['type']}:{task['query']}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        # 実際の処理はここで行う
        result = None  # 実際の処理結果
        self.cache[cache_key] = result
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポートの生成"""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = list(self.metrics_history)[-10:]  # 最近の10件
        
        return {
            "average_response_time": np.mean([m.response_time for m in recent_metrics]),
            "average_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
            "average_cpu_usage": np.mean([m.cpu_usage for m in recent_metrics]),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "total_requests": len(self.metrics_history),
            "timestamp": datetime.now().isoformat()
        }
    
    def optimize_streaming(self, model: str, chunk_size: int = None) -> Dict[str, Any]:
        """ストリーミング出力の最適化"""
        if model == "gemma3:latest":
            return {
                "chunk_size": chunk_size or 64,
                "buffer_size": 1024,
                "timeout": 30
            }
        return {}
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        self.thread_pool.shutdown()
        self.cache.clear()
        self.metrics_history.clear()

# グローバルなパフォーマンスモニターインスタンス
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """パフォーマンスモニターのインスタンスを取得"""
    return performance_monitor 