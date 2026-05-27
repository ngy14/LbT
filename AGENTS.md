# Repository Guidelines

## Language

- 回答・説明は日本語で行う。
- コード内コメントは既存コードの言語に合わせる。

## Project Structure & Module Organization

このリポジトリは、Learning by Teaching 用の Streamlit アプリです。

- `lbt_programming_binarysearch/app.py`: メインアプリ。UI、OpenAI API 呼び出し、対話制御、知識状態管理、JSON ログ出力を含みます。
- `lbt_programming_binarysearch/requirements.txt`: Python の依存関係。
- `lbt_programming_binarysearch/.env.example`: API キーとモデル ID の設定例。
- `lbt_programming_binarysearch/.streamlit/config.toml`: Streamlit 設定。
- `LbT.py` とルートの `README.md`: 現在はプレースホルダーです。

専用の `tests/` ディレクトリはまだありません。テストを追加する場合は `lbt_programming_binarysearch/tests/` に配置してください。

## Build, Test, and Development Commands

特記がない限り、コマンドは `lbt_programming_binarysearch/` で実行します。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

仮想環境を作成し、依存関係をインストールします。

```bash
streamlit run app.py
```

アプリを `http://localhost:8501` で起動します。

```bash
python -m py_compile app.py
```

Python の構文チェックを行います。

## Coding Style & Naming Conventions

Python 3 を使用し、インデントは 4 スペースにしてください。明確な利点がない限り、既存の `app.py` 中心の構成に合わせます。関数名・変数名は `call_gpt`、`op_extract`、`judge_answer_quality` のように、説明的な snake_case を使います。

ユーザーに見える教育フローに関わるプロンプトや UI 文言は日本語を維持してください。コメントは簡潔で有用なものにし、依頼と無関係な大規模リファクタリングは避けます。

## Testing Guidelines

現在、自動テストフレームワークは設定されていません。変更後は最低限、以下で確認してください。

```bash
python -m py_compile app.py
streamlit run app.py
```

テストを追加する場合は `pytest` を使い、`lbt_programming_binarysearch/tests/` に `test_*.py` 形式で配置します。純粋関数、JSON 抽出、質問文の正規化、知識状態更新を優先してテストしてください。

## Commit & Pull Request Guidelines

Git 履歴は初期コミットのみで、確立済みの規約はありません。コミットメッセージは短く、命令形にしてください。

```text
Add OpenAI env loader
Update Streamlit setup docs
```

Pull Request には、変更概要、手動確認手順、UI 変更がある場合はスクリーンショットを含めてください。モデル、プロンプト、ログ形式の変更は明示してください。

## Security & Configuration Tips

## Security rules

- `.env`, secret keys, API tokens, private keys, credentials, customer data を読まない・表示しない・変更しない。
- 外部通信が必要なコマンドを実行する前に理由を説明して承認を求める。
- 勝手に `git push`、本番デプロイ、DBマイグレーション、課金系APIの実行をしない。
- 機密情報が必要な場合は、値そのものではなく変数名や設定方法だけを説明する。
- 実際の API キーはコミットしないでください。秘密情報は `lbt_programming_binarysearch/.env` または `.streamlit/secrets.toml` に置きます。どちらも git から除外されています。`.env.example` には `OPENAI_API_KEY=your_openai_api_key` のようなプレースホルダーだけを書いてください。
