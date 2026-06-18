# LbT Programming Binary Search

プログラミング分野の二分探索を題材にした Learning by Teaching 用 Streamlit アプリです。

## まず実行する

前提:

- Python 3.10 以上
- OpenAI API キー

リポジトリのルートから実行する場合:

```bash
cd lbt_programming_binarysearch
```

仮想環境を作成して依存関係を入れます。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env.example` をコピーして `.env` を作ります。

```bash
cp .env.example .env
```

`.env` を開き、`OPENAI_API_KEY` に自分の API キーを設定します。

```text
OPENAI_API_KEY=sk-...
```

アプリを起動します。

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。

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

`.env.example` をコピーして `.env` を作成し、APIキーを設定してください。`.env` は `.gitignore` に含まれているため、gitには追加されません。

```bash
cp .env.example .env
```

`.env` の中身を編集します。

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

## 動作確認

構文チェック:

```bash
python -m py_compile app.py
```

アプリ起動:

```bash
streamlit run app.py
```

初回メッセージとして AlgoBo が「二分探索って何ですか？どうしてソートが必要なんですか！」と表示されれば起動できています。

## よくあるエラー

`OpenAI APIキーが見つかりません。OPENAI_API_KEY を設定してください。` と表示される場合は、`.env` が存在し、`OPENAI_API_KEY` が設定されているか確認してください。

`streamlit: command not found` と表示される場合は、仮想環境を有効化してから依存関係をインストールしてください。

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`python: command not found` と表示される環境では、代わりに `python3` を使ってください。

## uv を使う場合

`uv` が入っている環境では、以下でも実行できます。

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app.py
```

## 注意

元のColabノートブックにあった `!pip install`、`%%writefile`、`!streamlit run`、`cloudflared` 実行セルは削除・分離しています。
また、認証情報はgitに含めない前提で `.gitignore` に `.env` と `.streamlit/secrets.toml` を追加しています。
