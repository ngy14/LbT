from __future__ import annotations

import datetime
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from openai import OpenAI


APP_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = APP_DIR / "frontend" / "dist"

EXIT_COMMANDS = {"exit", "q", "quit", "終了", "おわり", "終わり"}
INITIAL_MESSAGE = "二分探索って何ですか？どうしてソートが必要なんですか！"
QUESTION_EVERY_N_MESSAGES = 3


def load_env_file(path: Path = APP_DIR / ".env") -> None:
    if not path.exists():
        return

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-5.2")
OPENAI_MODEL_ID_GEN = os.getenv("OPENAI_MODEL_ID_GEN", OPENAI_MODEL_ID)
OPENAI_MODEL_ID_CLS = os.getenv("OPENAI_MODEL_ID_CLS", "gpt-5-nano")
OPENAI_MODEL_ID_TQG = os.getenv("OPENAI_MODEL_ID_TQG", OPENAI_MODEL_ID_GEN)
OPENAI_MODEL_ID_PARA = os.getenv("OPENAI_MODEL_ID_PARA", OPENAI_MODEL_ID_CLS)

MODEL_IDS = {
    "GEN": OPENAI_MODEL_ID_GEN,
    "CLS": OPENAI_MODEL_ID_CLS,
    "TQG": OPENAI_MODEL_ID_TQG,
    "PARA": OPENAI_MODEL_ID_PARA,
}


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _to_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def call_gpt(
    client: OpenAI | None,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = 0.0,
    max_tokens: int = 1024,
    model_id: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    if not client:
        return None, None

    model_id = model_id or OPENAI_MODEL_ID_GEN
    params: dict[str, Any] = {
        "model": model_id,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": max_tokens,
    }
    if temperature is not None:
        params["temperature"] = temperature

    try:
        try:
            response = client.responses.create(**params)
        except Exception as e:
            if "temperature" not in params or "temperature" not in str(e).lower():
                raise
            params.pop("temperature", None)
            response = client.responses.create(**params)

        meta = {
            "model_id": model_id,
            "status": getattr(response, "status", None),
            "usage": _to_plain_data(getattr(response, "usage", None)),
        }
        return getattr(response, "output_text", "") or "", meta
    except Exception as e:
        return f"OpenAI API Error: {e}", {"model_id": model_id, "error": str(e)}


def call_role(
    client: OpenAI | None,
    role: Literal["GEN", "CLS", "TQG", "PARA"],
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = 0.0,
    max_tokens: int = 1024,
) -> tuple[str | None, dict[str, Any] | None]:
    return call_gpt(
        client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model_id=MODEL_IDS.get(role, OPENAI_MODEL_ID_GEN),
    )


def clamp_sentences_ja(text: str, max_sentences: int = 2, max_chars: int = 500) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[。！？.!?])|\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    out = "".join(parts[:max_sentences]) if parts else text.strip()
    out = out.strip()
    if len(out) > max_chars:
        out = out[:max_chars].strip()
    return out


def normalize_question(text: str, max_chars: int = 240) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    qm = [m.start() for m in re.finditer(r"[？?]", t)]
    if qm:
        t = t[: qm[-1] + 1].strip()
    elif t:
        t += "？"
    if len(t) > max_chars:
        t = t[-max_chars:].lstrip("、。．.!！ ")
    if not t or re.fullmatch(r"[？?]+", t):
        return "ソートされていない配列だと、どんな間違いが起きますか？"
    return t


def extract_json_block(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty")
    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right <= left:
        raise ValueError("no json object found")
    return json.loads(text[left : right + 1])


def op_extract(client: OpenAI | None, conversation: str) -> list[str]:
    system_prompt = """
あなたは、会話の内容を分析し、AlgoBoの知識状態に追加すべき新しい知識を抽出する専門家です。

厳守:
- JSON以外を出力しない
- CONVERSATIONに登場する文言をそのまま抽出する
- CONVERSATIONに無い知識を推測して追加しない
- 知識がない場合は {"knowledge": []} を返す

出力:
{
  "knowledge": ["抽出された自然言語の知識またはコード"]
}
"""
    user_prompt = f"CONVERSATION:\n{conversation}"
    res, _ = call_role(client, "GEN", system_prompt, user_prompt, temperature=0.0, max_tokens=1024)
    try:
        obj = extract_json_block(res or "")
        knowledge = obj.get("knowledge", [])
        if not isinstance(knowledge, list):
            return []
        return [str(item).strip() for item in knowledge if str(item).strip()]
    except Exception:
        return []


def op_update(client: OpenAI | None, current_state: dict[str, list[str]], new_knowledge: list[str]) -> dict[str, list[str]]:
    if not new_knowledge:
        return current_state

    current_knowledge_str = json.dumps(current_state, ensure_ascii=False, indent=2)
    new_knowledge_str = "\n".join([f"- {item}" for item in new_knowledge])
    system_prompt = """
あなたは知識統合の専門家です。
現在の知識状態と新しい知識を統合し、JSONだけを出力してください。

ルール:
- 重複はまとめる
- 矛盾がある場合は新しい知識を優先する
- 自然言語は facts、コードは code_implementation に入れる

出力:
{
  "facts": ["..."],
  "code_implementation": ["..."]
}
"""
    user_prompt = f"[KNOWLEDGE]\n{current_knowledge_str}\n\n[NEW KNOWLEDGE]\n{new_knowledge_str}"
    res, _ = call_role(client, "GEN", system_prompt, user_prompt, temperature=0.0, max_tokens=1024)
    try:
        updated = extract_json_block(res or "")
        return {
            "facts": list(updated.get("facts", [])),
            "code_implementation": list(updated.get("code_implementation", [])),
        }
    except Exception:
        merged = current_state.copy()
        merged["facts"] = list(dict.fromkeys(merged.get("facts", []) + new_knowledge))
        return merged


def op_check_consistency(
    client: OpenAI | None,
    current_state: dict[str, list[str]],
    new_knowledge: list[str],
) -> dict[str, str]:
    if not new_knowledge:
        return {"verdict": "NO_NEW_KNOWLEDGE", "summary": "", "conflict_with": ""}

    current_items = current_state.get("facts", []) + current_state.get("code_implementation", [])
    if not current_items:
        return {"verdict": "NO_PRIOR_KNOWLEDGE", "summary": "", "conflict_with": ""}

    current_text = "\n".join(current_items)
    new_text = "\n".join(new_knowledge)

    says_unsorted_always_ok = (
        ("ソートされていなくても" in new_text or "並んでいなくても" in new_text or "整列されていなくても" in new_text)
        and ("常に" in new_text or "いつでも" in new_text or "必ず" in new_text)
        and ("正しく" in new_text or "探せ" in new_text or "探索" in new_text)
    )
    prior_depends_on_order = (
        ("並んでいる" in current_text or "ソート" in current_text or "整列" in current_text)
        and ("左" in current_text or "右" in current_text or "半分" in current_text or "真ん中" in current_text or "中央" in current_text)
    )
    if says_unsorted_always_ok and prior_depends_on_order:
        return {
            "verdict": "CONFLICT",
            "summary": "未ソートでも常に正しく探せるという説明は、既存知識と矛盾します。",
            "conflict_with": current_items[0],
        }

    says_sort_needed_for_direction = (
        ("ソートが必要" in new_text or "並んでいる必要" in new_text or "整列が必要" in new_text)
        and ("左" in new_text or "右" in new_text)
        and ("判断" in new_text or "決め" in new_text or "行く" in new_text)
    )
    if says_sort_needed_for_direction and prior_depends_on_order:
        return {"verdict": "CONSISTENT", "summary": "", "conflict_with": ""}

    system_prompt = """
あなたはAlgoBoの知識状態を点検する評価者です。
CURRENT KNOWLEDGE と LATEST KNOWLEDGE を比較し、明確な矛盾があるか判定してください。

ルール:
- 言い換え、詳細化、補足、抽象度の違いは CONSISTENT
- 明確に両立しない内容だけ CONFLICT
- 判断に迷う場合は UNCLEAR

JSONのみ:
{
  "verdict": "CONSISTENT" | "CONFLICT" | "UNCLEAR",
  "summary": "短い判定理由",
  "conflict_with": "矛盾する既存知識の短い抜粋。なければ空文字"
}
"""
    user_prompt = (
        f"[CURRENT KNOWLEDGE]\n{json.dumps(current_state, ensure_ascii=False, indent=2)}"
        f"\n\n[LATEST KNOWLEDGE]\n{json.dumps(new_knowledge, ensure_ascii=False, indent=2)}"
    )
    res, _ = call_role(client, "CLS", system_prompt, user_prompt, temperature=0.0, max_tokens=256)
    try:
        obj = extract_json_block(res or "")
        verdict = str(obj.get("verdict", "UNCLEAR")).upper()
        if verdict not in {"CONSISTENT", "CONFLICT", "UNCLEAR"}:
            verdict = "UNCLEAR"
        return {
            "verdict": verdict,
            "summary": str(obj.get("summary", "")),
            "conflict_with": str(obj.get("conflict_with", "")),
        }
    except Exception:
        return {"verdict": "UNCLEAR", "summary": "", "conflict_with": ""}


def op_compose(
    client: OpenAI | None,
    latest_knowledge: list[str],
    context: str,
    consistency_check: dict[str, str],
) -> tuple[str, dict[str, Any] | None]:
    if not latest_knowledge:
        return "二分探索についてもっと詳しく説明してもらえませんか？", None

    system_prompt = """
あなたはAlgoBoという、プログラミングを学ぶ1年目の学生です。
回答は日本語で、Tutorに教えてもらった内容を確認する口調にしてください。

制約:
- 基本は1文だけ
- LATEST KNOWLEDGEに書かれた直前のTutor発話内容だけを確認する
- LATEST KNOWLEDGEにない定義、手順、理由、例、補足知識を追加しない
- 「〜なんだよ」「〜です！」のように先生としてまとめ直す口調は禁止
- CONSISTENCY CHECK の verdict が CONFLICT の場合だけ、2文目でどちらで理解すべきか質問する
- verdict が CONFLICT 以外の場合は、矛盾確認について触れない
"""
    user_prompt = (
        f"[CONVERSATION CONTEXT]\n{context}\n\n"
        f"[LATEST KNOWLEDGE]\n{json.dumps(latest_knowledge, ensure_ascii=False, indent=2)}\n\n"
        f"[CONSISTENCY CHECK]\n{json.dumps(consistency_check, ensure_ascii=False, indent=2)}"
    )
    res, meta = call_role(client, "GEN", system_prompt, user_prompt, temperature=0.2, max_tokens=512)
    if not res:
        return "二分探索についてもっと詳しく説明してもらえませんか？", meta
    return clamp_sentences_ja(res, 2), meta


def generate_question(client: OpenAI | None, context: str, past_questions: list[str]) -> tuple[str, dict[str, Any] | None]:
    past_q = "\n".join([f"- {q}" for q in past_questions])
    system_prompt = f"""
あなたはAlgoBoという、プログラミングを学ぶ熱心な学生です。
Tutorの理解を深めるため、初学者に自然な質問を1つだけ出力してください。

ルール:
- 最大1文
- 「？」で終える
- 説明文は禁止
- 「不変条件」「単調性」のような専門語は禁止
- 「どんな」と「なぜ」を混ぜた不自然な文は禁止
- 過去の質問と重複しない

[過去の質問]
{past_q}
"""
    user_prompt = f"[現在の会話文脈]\n{context}"
    res, meta = call_role(client, "TQG", system_prompt, user_prompt, temperature=0.7, max_tokens=256)
    return normalize_question(res or "ソートされていない配列だと、どんな間違いが起きますか？"), meta


@dataclass
class SessionState:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=lambda: [{"role": "assistant", "content": INITIAL_MESSAGE}])
    knowledge_state: dict[str, list[str]] = field(default_factory=lambda: {"facts": [], "code_implementation": []})
    msg_count: int = 0
    mode: str = "HELP_RECEIVER"
    past_questions: list[str] = field(default_factory=list)
    pending_question_after_conflict: bool = False
    turns: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


sessions: dict[str, SessionState] = {}

app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})


def json_error(message: str, status: int):
    response = jsonify({"error": message})
    response.status_code = status
    return response


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "has_openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
            "model_ids": MODEL_IDS,
        }
    )


@app.post("/api/session")
def create_session():
    session_id = uuid.uuid4().hex
    state = SessionState(session_id=session_id)
    sessions[session_id] = state
    return jsonify({"session_id": session_id, "messages": state.messages, "mode": state.mode})


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    message = str(payload.get("message", "")).strip()

    state = sessions.get(session_id)
    if not state:
        return json_error("session not found", 404)
    if not message:
        return json_error("message is empty", 400)

    if message.lower() in EXIT_COMMANDS:
        assistant = "了解！ここで終了にします。"
        state.messages.append({"role": "user", "content": message})
        state.messages.append({"role": "assistant", "content": assistant})
        return jsonify(
            {
                "session_id": state.session_id,
                "assistant": assistant,
                "messages": state.messages,
                "mode": state.mode,
                "knowledge_state": state.knowledge_state,
                "ended": True,
            }
        )

    client = get_openai_client()
    if not client:
        return json_error("OPENAI_API_KEY is not configured", 503)

    state.messages.append({"role": "user", "content": message})
    context = "\n".join([m["content"] for m in state.messages[-4:]])
    new_knowledge = op_extract(client, message)
    consistency_check = op_check_consistency(client, state.knowledge_state, new_knowledge)

    state.msg_count += 1
    msg_count_after = state.msg_count

    if consistency_check.get("verdict") == "CONFLICT":
        assistant, llm_meta = op_compose(client, new_knowledge, context, consistency_check)
        state.mode = "HELP_RECEIVER"
        if msg_count_after % QUESTION_EVERY_N_MESSAGES == 0:
            state.pending_question_after_conflict = True
    else:
        if new_knowledge:
            state.knowledge_state = op_update(client, state.knowledge_state, new_knowledge)

        should_question = state.pending_question_after_conflict or (msg_count_after % QUESTION_EVERY_N_MESSAGES == 0)
        if should_question:
            assistant, llm_meta = generate_question(client, context, state.past_questions)
            state.mode = "QUESTIONER"
            state.past_questions.append(assistant)
            state.pending_question_after_conflict = False
        else:
            assistant, llm_meta = op_compose(client, new_knowledge, context, consistency_check)
            state.mode = "HELP_RECEIVER"

    state.messages.append({"role": "assistant", "content": assistant})
    state.turns.append(
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "tutor": message,
            "algobo": assistant,
            "mode": state.mode,
            "knowledge_delta": new_knowledge,
            "consistency_check": consistency_check,
            "llm_meta": llm_meta,
            "pending_question_after_conflict": state.pending_question_after_conflict,
        }
    )

    return jsonify(
        {
            "session_id": state.session_id,
            "assistant": assistant,
            "messages": state.messages,
            "mode": state.mode,
            "knowledge_state": state.knowledge_state,
            "consistency_check": consistency_check,
            "ended": False,
        }
    )


@app.get("/")
def serve_index():
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return json_error("React frontend is not built. Run npm install && npm run build in frontend/.", 404)


@app.get("/<path:path>")
def serve_frontend(path: str):
    target = FRONTEND_DIST / path
    if target.exists():
        return send_from_directory(FRONTEND_DIST, path)
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return json_error("React frontend is not built. Run npm install && npm run build in frontend/.", 404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
