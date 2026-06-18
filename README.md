# LbT

Learning by Teaching（LbT）形式の対話システムを管理するリポジトリです。プログラミング分野の二分探索を題材にしたアプリを含んでいます。

## アプリ構成

```text
.
├── lbt_programming_binarysearch/
│   └── 既存の Streamlit 版
└── lbt_programming_binarysearch_web/
    └── 新しい Flask + React 版
```

- `lbt_programming_binarysearch/`: 既存の Streamlit アプリです。これまでの単一 Python アプリを残しています。
- `lbt_programming_binarysearch_web/`: バックエンドとフロントエンドを分けた新しい構成です。バックエンドは Flask、フロントエンドは React です。

## Flask + React 版を実行する

前提:

- Docker / Docker Compose
- OpenAI API キー

```bash
git clone <このリポジトリのURL>
cd LbT/lbt_programming_binarysearch_web
cp .env.example .env
```

`.env` を開き、`OPENAI_API_KEY` に自分の API キーを設定します。

```text
OPENAI_API_KEY=sk-...
```

Docker で起動します。

```bash
docker compose up --build
```

ブラウザで `http://localhost:5000` を開きます。

## Streamlit 版を実行する

```bash
cd lbt_programming_binarysearch

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

`.env` に `OPENAI_API_KEY` を設定してから起動します。

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。

## 確認コマンド

Streamlit 版:

```bash
cd lbt_programming_binarysearch
python -m py_compile app.py
```

Flask + React 版:

```bash
cd lbt_programming_binarysearch_web
python -m py_compile backend/app.py
docker compose build
```

## よくあるエラー

`OPENAI_API_KEY` が見つからない場合は、各アプリフォルダの `.env` が存在し、`OPENAI_API_KEY` が設定されているか確認してください。

`ModuleNotFoundError` が出る場合は、仮想環境を有効化してから `pip install -r requirements.txt` を実行してください。

`docker: command not found` と表示される場合は、Docker Desktop または Docker Engine をインストールしてください。
