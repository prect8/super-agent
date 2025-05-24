import streamlit as st
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agents import SuperAgent
from config import Config
from data_analysis import DataAnalyzer

# 基本設定
config = Config()
st.set_page_config(**config.PAGE_CONFIG)

# Ollamaモデル設定
MODEL_CONFIGS = {
    "deepseek-r1:14b": {
        "name": "DeepSeek R1 14B",
        "description": "高度な推論能力を持つモデル。複雑な分析と問題解決に最適。",
        "recommended_tasks": ["reasoning", "analysis", "research"],
        "max_tokens": 4096,
        "temperature_range": (0.0, 1.5)
    },
    "gemma3:latest": {
        "name": "Gemma 3",
        "description": "創造的なタスクに特化したモデル。文章作成と創作に最適。",
        "recommended_tasks": ["creative", "writing", "chat"],
        "max_tokens": 2048,
        "temperature_range": (0.0, 2.0)
    },
    "llava:latest": {
        "name": "LLaVA",
        "description": "マルチモーダルモデル。画像理解とテキスト生成の両方に対応。",
        "recommended_tasks": ["vision", "analysis", "chat"],
        "max_tokens": 2048,
        "temperature_range": (0.0, 1.0)
    },
    "llava:13b": {
        "name": "LLaVA 13B",
        "description": "大型のマルチモーダルモデル。詳細な画像分析に最適。",
        "recommended_tasks": ["vision", "analysis", "reasoning"],
        "max_tokens": 4096,
        "temperature_range": (0.0, 1.0)
    },
    "moondream:latest": {
        "name": "Moondream",
        "description": "軽量で高速な画像理解モデル。基本的な画像処理に最適。",
        "recommended_tasks": ["vision", "chat"],
        "max_tokens": 1024,
        "temperature_range": (0.0, 1.0)
    },
    "qwen2.5:latest": {
        "name": "Qwen 2.5",
        "description": "高性能な多言語モデル。翻訳と要約に優れた性能。",
        "recommended_tasks": ["translation", "summarization", "chat"],
        "max_tokens": 8192,
        "temperature_range": (0.0, 1.5)
    }
}

# タスクタイプの定義
TASK_TYPES = {
    "chat": "一般的な会話",
    "creative": "創造的な文章生成",
    "analysis": "データ分析と解釈",
    "research": "リサーチと情報収集",
    "summarization": "要約と整理",
    "translation": "翻訳",
    "reasoning": "論理的推論",
    "writing": "文章作成",
    "vision": "画像理解・処理"
}

# 履歴管理用のディレクトリ
HISTORY_DIR = "chat_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)

# 統計データの保存用ディレクトリ
STATS_DIR = "usage_stats"
os.makedirs(STATS_DIR, exist_ok=True)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = SuperAgent()
if "processing" not in st.session_state:
    st.session_state.processing = False
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "deepseek-r1:14b"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "task_type" not in st.session_state:
    st.session_state.task_type = "chat"
if "data_analyzer" not in st.session_state:
    st.session_state.data_analyzer = DataAnalyzer()
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_requests": 0,
        "model_usage": {},
        "response_times": [],
        "errors": 0,
        "daily_usage": {},
        "hourly_usage": {}
    }

def get_recommended_models(task_type: str) -> List[str]:
    """タスクタイプに基づいて推奨モデルを取得"""
    recommended = []
    for model_id, config in MODEL_CONFIGS.items():
        if task_type in config["recommended_tasks"]:
            recommended.append(model_id)
    return recommended

def update_model_settings():
    """モデル設定を更新"""
    st.session_state.agent.update_model_settings(
        model=st.session_state.selected_model,
        temperature=st.session_state.temperature
    )

def save_conversation(messages: List[Dict[str, Any]], filename: str):
    """会話をJSONファイルとして保存"""
    try:
        data = {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages
        }
        filepath = os.path.join(HISTORY_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存に失敗しました: {e}")
        return False

def load_conversation(filename: str) -> Optional[List[Dict[str, Any]]]:
    """会話をJSONファイルから読み込み"""
    try:
        filepath = os.path.join(HISTORY_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])
    except Exception as e:
        st.error(f"読み込みに失敗しました: {e}")
        return None

def search_messages(query: str) -> List[Dict[str, Any]]:
    """メッセージを検索"""
    if not query:
        return st.session_state.messages
    
    query = query.lower()
    results = []
    for message in st.session_state.messages:
        content = message.get("content", "").lower()
        if query in content:
            results.append(message)
    return results

def delete_message(index: int):
    """メッセージを削除"""
    if 0 <= index < len(st.session_state.messages):
        st.session_state.messages.pop(index)
        st.rerun()

def save_stats():
    """統計データを保存"""
    try:
        filepath = os.path.join(STATS_DIR, "usage_stats.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(st.session_state.stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"統計データの保存に失敗しました: {e}")

def load_stats():
    """統計データを読み込み"""
    try:
        filepath = os.path.join(STATS_DIR, "usage_stats.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                st.session_state.stats = json.load(f)
    except Exception as e:
        st.error(f"統計データの読み込みに失敗しました: {e}")

def update_stats(result: Dict[str, Any], response_time: float, is_error: bool = False):
    """統計データを更新"""
    stats = st.session_state.stats
    
    # 基本統計
    stats["total_requests"] += 1
    if is_error:
        stats["errors"] += 1
    
    # モデル使用回数
    model = result.get("model_info", {}).get("model", "unknown")
    stats["model_usage"][model] = stats["model_usage"].get(model, 0) + 1
    
    # 応答時間
    stats["response_times"].append(response_time)
    
    # 日時別使用量
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H:00")
    
    stats["daily_usage"][date_str] = stats["daily_usage"].get(date_str, 0) + 1
    stats["hourly_usage"][hour_str] = stats["hourly_usage"].get(hour_str, 0) + 1
    
    # データを保存
    save_stats()

def create_usage_graph(data: Dict[str, int], title: str, xaxis_title: str):
    """使用量グラフを作成"""
    df = pd.DataFrame(list(data.items()), columns=["time", "count"])
    df = df.sort_values("time")
    
    fig = px.bar(
        df,
        x="time",
        y="count",
        title=title,
        labels={"time": xaxis_title, "count": "使用回数"}
    )
    
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title="使用回数",
        showlegend=False
    )
    
    return fig

def create_model_usage_pie():
    """モデル使用回数の円グラフを作成"""
    model_usage = st.session_state.stats["model_usage"]
    if not model_usage:
        return None
    
    fig = px.pie(
        values=list(model_usage.values()),
        names=list(model_usage.keys()),
        title="モデル別使用回数"
    )
    
    return fig

def show_statistics():
    """統計情報を表示"""
    stats = st.session_state.stats
    
    # 基本統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総リクエスト数", stats["total_requests"])
    
    with col2:
        error_rate = (stats["errors"] / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0
        st.metric("エラー率", f"{error_rate:.1f}%")
    
    with col3:
        avg_response_time = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
        st.metric("平均応答時間", f"{avg_response_time:.1f}秒")
    
    with col4:
        unique_models = len(stats["model_usage"])
        st.metric("使用モデル数", unique_models)
    
    # グラフ表示
    col1, col2 = st.columns(2)
    
    with col1:
        # 日別使用量
        daily_fig = create_usage_graph(
            stats["daily_usage"],
            "日別使用量",
            "日付"
        )
        st.plotly_chart(daily_fig, use_container_width=True)
        
        # 時間別使用量
        hourly_fig = create_usage_graph(
            stats["hourly_usage"],
            "時間別使用量",
            "時間"
        )
        st.plotly_chart(hourly_fig, use_container_width=True)
    
    with col2:
        # モデル使用回数
        model_fig = create_model_usage_pie()
        if model_fig:
            st.plotly_chart(model_fig, use_container_width=True)
        
        # 応答時間の分布
        if stats["response_times"]:
            response_times_df = pd.DataFrame(stats["response_times"], columns=["response_time"])
            fig = px.histogram(
                response_times_df,
                x="response_time",
                title="応答時間の分布",
                labels={"response_time": "応答時間（秒）", "count": "回数"}
            )
            st.plotly_chart(fig, use_container_width=True)

def analyze_data(data: Union[pd.DataFrame, str, List[Dict[str, Any]]], analysis_type: str = "basic"):
    """データ分析を実行
    
    Args:
        data: 分析対象のデータ
        analysis_type: 分析タイプ ('basic', 'correlation', 'visualization')
    """
    analyzer = st.session_state.data_analyzer
    analyzer.load_data(data)
    
    results = {}
    if analysis_type == "basic":
        results = analyzer.basic_statistics()
    elif analysis_type == "correlation":
        results = analyzer.correlation_analysis()
    
    return results

def show_data_analysis_interface():
    """データ分析インターフェースを表示"""
    st.subheader("データ分析")
    
    # データ入力
    data_input = st.text_area("データを入力（JSON形式）", height=200)
    
    if data_input:
        try:
            # データの解析
            analysis_type = st.selectbox(
                "分析タイプ",
                ["basic", "correlation", "visualization"]
            )
            
            if st.button("分析実行"):
                results = analyze_data(data_input, analysis_type)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.json(results)
                    
                    # 可視化の表示
                    if analysis_type == "visualization":
                        plot_type = st.selectbox(
                            "プロットタイプ",
                            ["scatter", "line", "bar", "histogram"]
                        )
                        if st.button("プロット生成"):
                            plot_result = st.session_state.data_analyzer.create_visualization(
                                plot_type=plot_type
                            )
                            if "plot" in plot_result:
                                st.plotly_chart(json.loads(plot_result["plot"]))
                            else:
                                st.error(plot_result.get("error", "プロットの生成に失敗しました"))
        
        except Exception as e:
            st.error(f"データの解析に失敗しました: {e}")

def main():
    st.title("🤖 スーパーエージェント")
    st.markdown("マルチモーダル + ディープリサーチ対応AIアシスタント")

    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 統計タブ
        tab1, tab2 = st.tabs(["モデル設定", "使用統計"])
        
        with tab1:
            # モデル設定
            st.subheader("🤖 モデル設定")
            
            # タスクタイプ選択
            task_type = st.selectbox(
                "タスクタイプ",
                options=list(TASK_TYPES.keys()),
                format_func=lambda x: TASK_TYPES[x],
                key="task_type"
            )
            
            # 推奨モデル表示
            recommended_models = get_recommended_models(task_type)
            if recommended_models:
                st.info("推奨モデル: " + ", ".join(MODEL_CONFIGS[m]["name"] for m in recommended_models))
            
            # モデル選択
            selected_model = st.selectbox(
                "モデル",
                options=list(MODEL_CONFIGS.keys()),
                format_func=lambda x: MODEL_CONFIGS[x]["name"],
                key="selected_model"
            )
            
            # モデル詳細表示
            model_config = MODEL_CONFIGS[selected_model]
            with st.expander("モデル詳細"):
                st.write(f"**説明**: {model_config['description']}")
                st.write(f"**最大トークン数**: {model_config['max_tokens']:,}")
                st.write("**推奨タスク**:")
                for task in model_config["recommended_tasks"]:
                    st.write(f"- {TASK_TYPES[task]}")
            
            # 温度設定
            temp_range = model_config["temperature_range"]
            temperature = st.slider(
                "温度 (創造性)",
                min_value=temp_range[0],
                max_value=temp_range[1],
                value=st.session_state.temperature,
                step=0.1,
                help="高いほど創造的、低いほど決定論的な応答になります",
                key="temperature"
            )
            
            # 設定適用ボタン
            if st.button("⚡ 設定を適用"):
                update_model_settings()
                st.success("モデル設定を更新しました")
            
            st.divider()
            
            # 会話記憶設定
            st.subheader("🧠 会話記憶設定")
            
            # エージェントの記憶状況表示
            if hasattr(st.session_state.agent, 'conversation_history'):
                agent_memory_count = len(st.session_state.agent.conversation_history)
                st.metric("エージェント記憶数", agent_memory_count)
                
                # 記憶の内容を表示
                if agent_memory_count > 0:
                    with st.expander("🔍 記憶内容を確認"):
                        for i, msg in enumerate(st.session_state.agent.conversation_history[-6:]):  # 最新6件
                            role_emoji = "👤" if msg["role"] == "user" else "🤖" if msg["role"] == "assistant" else "⚙️"
                            content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            st.write(f"{role_emoji} **{msg['role']}**: {content_preview}")
                
                # 記憶クリアボタン
                if st.button("🧹 エージェント記憶をクリア"):
                    st.session_state.agent.conversation_history = []
                    st.success("エージェントの記憶をクリアしました")
                    st.rerun()
            
            st.divider()
            
            # 検索機能
            st.subheader("🔍 履歴検索")
            search_query = st.text_input("キーワード検索", key="search_input")
            if search_query:
                st.session_state.search_query = search_query
            
            # 統計
            message_count = len(st.session_state.messages)
            st.metric("対話回数", message_count // 2)
            
            # 履歴管理
            st.subheader("💾 履歴管理")
            
            # 保存
            if message_count > 0:
                save_name = st.text_input("保存名", value=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                if st.button("💾 会話を保存"):
                    if save_conversation(st.session_state.messages, f"{save_name}.json"):
                        st.success("保存しました")
            
            # 読み込み
            st.subheader("📂 保存済み会話")
            saved_files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
            if saved_files:
                selected_file = st.selectbox("読み込む会話を選択", saved_files)
                if st.button("📖 会話を読み込み"):
                    messages = load_conversation(selected_file)
                    if messages:
                        st.session_state.messages = messages
                        st.success("読み込みました")
                        st.rerun()
            
            # エクスポート/インポート
            st.subheader("📤 エクスポート/インポート")
            
            # エクスポート
            if message_count > 0:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "message_count": message_count,
                    "messages": st.session_state.messages
                }
                st.download_button(
                    "📤 JSONエクスポート",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # インポート
            uploaded_file = st.file_uploader("📥 JSONインポート", type=["json"])
            if uploaded_file:
                try:
                    data = json.load(uploaded_file)
                    if st.button("📥 インポート実行"):
                        if "messages" in data:
                            st.session_state.messages = data["messages"]
                            st.success("インポートしました")
                            st.rerun()
                except Exception as e:
                    st.error(f"インポートに失敗しました: {e}")
            
            # リセット
            if st.button("🗑️ チャット履歴クリア"):
                st.session_state.messages = []
                st.rerun()

        with tab2:
            show_statistics()

    # チャット履歴表示
    messages_to_display = search_messages(st.session_state.search_query)
    for i, message in enumerate(messages_to_display):
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([20, 1])
            with col1:
                st.write(message["content"])
                if message.get("image"):
                    st.image(message["image"])
            with col2:
                if st.button("🗑️", key=f"delete_{i}"):
                    delete_message(i)
                    st.rerun()

    # 入力エリア
    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.chat_input(
            "メッセージを入力..." if not st.session_state.processing else "処理中...",
            disabled=st.session_state.processing
        )

    with col2:
        uploaded_file = st.file_uploader(
            "📷 画像アップロード",
            type=['png', 'jpg', 'jpeg'],
            disabled=st.session_state.processing
        )

    # 処理
    if user_input and not st.session_state.processing:
        try:
            st.session_state.processing = True
            start_time = datetime.now()
            
            # ユーザーメッセージ追加
            user_message = {"role": "user", "content": user_input}
            if uploaded_file:
                image = Image.open(uploaded_file)
                user_message["image"] = image
            st.session_state.messages.append(user_message)
            
            # AI応答生成（バックグラウンドで処理）
            with st.spinner("🤔 処理中..."):
                result = asyncio.run(
                    st.session_state.agent.process_query(
                        user_input,
                        image if uploaded_file else None,
                        conversation_history=st.session_state.messages
                    )
                )
            
            # 統計更新
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            update_stats(result, response_time)
            
            # アシスタントメッセージ保存
            assistant_message = {
                "role": "assistant",
                "content": result.get("response", "応答なし"),
                "task_type": result.get("task_type", "unknown"),
                "model_info": result.get("model_info", {}),
                "has_image": result.get("has_image", False),
                "research_data": result.get("research_data")
            }
            st.session_state.messages.append(assistant_message)
            
            st.session_state.processing = False
            st.rerun()
            
        except Exception as e:
            st.session_state.processing = False
            
            # エラー統計更新
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            update_stats({"model_info": {"model": "error"}}, response_time, is_error=True)
            
            # エラーメッセージ保存
            error_message = {
                "role": "assistant",
                "content": f"申し訳ありません。エラーが発生しました: {str(e)}",
                "task_type": "error",
                "model_info": {"model": "error"}
            }
            st.session_state.messages.append(error_message)
            st.rerun()

if __name__ == "__main__":
    main() 