import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, List, Dict, Any
import json

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.analysis_results = {}

    def load_data(self, data: Union[pd.DataFrame, str, List[Dict[str, Any]]]):
        """データを読み込む
        
        Args:
            data: pandas DataFrame、JSON文字列、または辞書のリスト
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            try:
                # JSON文字列を読み込む
                json_data = json.loads(data)
                self.data = pd.DataFrame(json_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string")
        elif isinstance(data, list):
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported data type")

    def basic_statistics(self) -> Dict[str, Any]:
        """基本的な統計情報を計算"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        stats = {
            "summary": self.data.describe().to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "data_types": self.data.dtypes.astype(str).to_dict()
        }
        self.analysis_results["basic_statistics"] = stats
        return stats

    def correlation_analysis(self) -> Dict[str, Any]:
        """相関分析を実行"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "Not enough numeric columns for correlation analysis"}
        
        corr_matrix = self.data[numeric_cols].corr()
        self.analysis_results["correlation"] = corr_matrix.to_dict()
        return corr_matrix.to_dict()

    def create_visualization(self, plot_type: str, **kwargs) -> Dict[str, Any]:
        """データの可視化を作成
        
        Args:
            plot_type: プロットの種類 ('scatter', 'line', 'bar', 'histogram')
            **kwargs: プロットのパラメータ
        """
        if self.data is None:
            raise ValueError("No data loaded")

        try:
            if plot_type == "scatter":
                fig = px.scatter(self.data, **kwargs)
            elif plot_type == "line":
                fig = px.line(self.data, **kwargs)
            elif plot_type == "bar":
                fig = px.bar(self.data, **kwargs)
            elif plot_type == "histogram":
                fig = px.histogram(self.data, **kwargs)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

            return {"plot": fig.to_json()}
        except Exception as e:
            return {"error": str(e)}

    def get_analysis_results(self) -> Dict[str, Any]:
        """分析結果を取得"""
        return self.analysis_results 