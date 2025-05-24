# AI Assistant with Data Analysis

マルチモーダル対応のAIアシスタントアプリケーション。チャット、画像分析、データ分析などの機能を提供します。

## 機能

- 🤖 マルチモーダルAIアシスタント
- 📊 データ分析と可視化
- 🖼️ 画像理解と処理
- 📝 会話履歴の管理
- 📈 使用統計の追跡

## 必要条件

- Python 3.8以上
- Ollama（ローカルでLLMを実行するため）

## インストール

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/ai-assistant.git
cd ai-assistant
```

2. 仮想環境を作成して有効化:
```bash
python -m venv venv
source venv/bin/activate  # Linuxの場合
venv\Scripts\activate     # Windowsの場合
```

3. 依存パッケージをインストール:
```bash
pip install -r requirements.txt
```

4. Ollamaをインストール:
- [Ollama公式サイト](https://ollama.ai/)からインストーラーをダウンロード
- 必要なモデルをダウンロード（例：`ollama pull deepseek-r1:14b`）

## 使用方法

1. アプリケーションを起動:
```bash
streamlit run app.py
```

2. ブラウザで http://localhost:8501 にアクセス

## データ分析機能

1. タスクタイプで「analysis」を選択
2. JSON形式でデータを入力
3. 分析タイプを選択（基本統計、相関分析、可視化）
4. 「分析実行」ボタンをクリック

## デプロイ

このアプリケーションはStreamlit Cloudでデプロイできます：

1. GitHubリポジトリをプッシュ
2. [Streamlit Cloud](https://streamlit.io/cloud)にアクセス
3. 新しいアプリを作成
4. リポジトリを選択
5. メインファイルとして`app.py`を指定

## ライセンス

MIT License

## 作者

Your Name 