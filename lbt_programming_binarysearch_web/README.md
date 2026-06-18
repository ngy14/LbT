# LbT Programming Binary Search Web

二分探索を題材にした Learning by Teaching 用アプリの Flask + React 版です。

- バックエンド: Flask / Python
- フロントエンド: React / Vite
- 既存の Streamlit 版は `../lbt_programming_binarysearch/` に残しています。

## まず実行する

前提:

- Docker / Docker Compose
- OpenAI API キー

```bash
cd lbt_programming_binarysearch_web
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

停止する場合:

```bash
docker compose down
```

## ローカル開発用に起動する

Docker を使わずに開発する場合は、Python と Node.js を使います。

前提:

- Python 3.10 以上
- Node.js 20 以上

```bash
cd lbt_programming_binarysearch_web
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

ターミナル1で Flask バックエンドを起動します。

```bash
cd lbt_programming_binarysearch_web
source .venv/bin/activate
python backend/app.py
```

ターミナル2で React フロントエンドを起動します。

```bash
cd lbt_programming_binarysearch_web/frontend
npm install
npm run dev
```

ブラウザで `http://localhost:5173` を開きます。

## ビルドして Flask から配信する

```bash
cd lbt_programming_binarysearch_web/frontend
npm install
npm run build

cd ..
source .venv/bin/activate
python backend/app.py
```

ブラウザで `http://localhost:5000` を開きます。

## ファイル構成

```text
.
├── backend/
│   └── app.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       └── styles.css
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## API

- `GET /api/health`: API キー設定とモデル設定の確認
- `POST /api/session`: 新しい会話セッションの作成
- `POST /api/chat`: Tutor の発話を送信し、AlgoBo の返答を取得

## 動作確認

```bash
python -m py_compile backend/app.py
```

Docker ビルド:

```bash
docker compose build
```

React 側は依存関係をインストールした後に確認します。

```bash
cd frontend
npm run build
```

## よくあるエラー

`OPENAI_API_KEY is not configured` と表示される場合は、`.env` が存在し、`OPENAI_API_KEY` が設定されているか確認してください。

`docker: command not found` と表示される場合は、Docker Desktop または Docker Engine をインストールしてください。

`ModuleNotFoundError: No module named 'flask'` と表示される場合は、仮想環境を有効化してから依存関係をインストールしてください。

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`npm: command not found` と表示される場合は、Node.js をインストールしてください。
