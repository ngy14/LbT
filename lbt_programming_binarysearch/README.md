# LbT Programming Binary Search

プログラミング分野の二分探索を題材にした Learning by Teaching 用 Streamlit アプリです。

## ファイル構成

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── .streamlit/
    └── config.toml
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate
pip install -r requirements.txt
```

## OpenAI APIキーの設定

`.env` を作成して APIキーを設定してください。`.env` は `.gitignore` に含まれているため、gitには追加されません。

```text
OPENAI_API_KEY=sk-...
```

環境変数で渡すこともできます。

```bash
export OPENAI_API_KEY="..."
```

必要に応じて利用モデルを変更できます。

```bash
export OPENAI_MODEL_ID="gpt-5.2"
export OPENAI_MODEL_ID_CLS="gpt-5-nano"
```

## 起動方法

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。

## 注意

元のColabノートブックにあった `!pip install`、`%%writefile`、`!streamlit run`、`cloudflared` 実行セルは削除・分離しています。
また、認証情報はgitに含めない前提で `.gitignore` に `.env` と `.streamlit/secrets.toml` を追加しています。
