# LbT

Learning by Teaching（LbT）形式の対話システムを管理するリポジトリです。現在は、プログラミング分野の二分探索を題材にした Streamlit アプリを含んでいます。

## まず実行する

前提:

- Python 3.10 以上
- OpenAI API キー

```bash
git clone <このリポジトリのURL>
cd LbT/lbt_programming_binarysearch

python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env
```

作成した `.env` を開き、`OPENAI_API_KEY` に自分の API キーを設定します。

```text
OPENAI_API_KEY=sk-...
```

起動します。

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。

## アプリ構成

```text
.
├── AGENTS.md
├── README.md
└── lbt_programming_binarysearch/
    ├── app.py
    ├── requirements.txt
    ├── .env.example
    └── .streamlit/
        └── config.toml
```

- `lbt_programming_binarysearch/`: プログラミング・二分探索向けの LbT アプリ。
- `AGENTS.md`: コントリビューター向けの作業ガイド。
- `LbT.py`: 現在は未使用のプレースホルダー。

## 概要

このアプリでは、ユーザーが Tutor として二分探索を教え、AI 学習者 `AlgoBo` が質問や応答を返します。Tutor の発話は分類され、指示偏重・説明しすぎ・情報不足などの傾向がある場合は Teaching Helper が改善を促します。

## セットアップ

```bash
cd lbt_programming_binarysearch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

この環境では `uv` を使う場合もあります。

```bash
uv venv .venv
uv pip install -r requirements.txt
```

## OpenAI API キー

`lbt_programming_binarysearch/.env.example` をコピーして `lbt_programming_binarysearch/.env` を作成し、API キーを設定します。

```bash
cd lbt_programming_binarysearch
cp .env.example .env
```

`.env` の中身を編集します。

```text
OPENAI_API_KEY=sk-...
```

必要に応じてモデルも変更できます。通常は `.env.example` のデフォルトのままで構いません。

`.env` は git 管理しないでください。

## 起動方法

```bash
cd lbt_programming_binarysearch
source .venv/bin/activate
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。

## 確認コマンド

```bash
python -m py_compile app.py
```

現在、自動テストは未整備です。変更後は構文チェックと Streamlit の手動起動確認を行ってください。

## よくあるエラー

`OpenAI APIキーが見つかりません。OPENAI_API_KEY を設定してください。` と表示される場合は、`lbt_programming_binarysearch/.env` が存在し、`OPENAI_API_KEY` が設定されているか確認してください。

`streamlit: command not found` と表示される場合は、仮想環境を有効化してから依存関係をインストールしてください。

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`python: command not found` と表示される環境では、代わりに `python3` を使ってください。
