import streamlit as st
from openai import OpenAI
import json
import re
from typing import Optional, List, Any, TypedDict, Dict
from collections import deque, Counter
import datetime
import os
import math
import unicodedata
import difflib


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as f:
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


# --- 設定と定数 ---
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-5.2")

# --- モデルIDを役割で分離（環境変数で差し替え可能に）---
OPENAI_MODEL_ID_GEN  = os.getenv("OPENAI_MODEL_ID_GEN",  OPENAI_MODEL_ID)       # 生成（AlgoBo応答、extract/update/retrieveなど）
OPENAI_MODEL_ID_CLS  = os.getenv("OPENAI_MODEL_ID_CLS",  "gpt-5-nano")          # 分類/判定（軽量）
OPENAI_MODEL_ID_TQG  = os.getenv("OPENAI_MODEL_ID_TQG",  OPENAI_MODEL_ID_GEN)   # Thinking Question Generator
OPENAI_MODEL_ID_PARA = os.getenv("OPENAI_MODEL_ID_PARA", OPENAI_MODEL_ID_CLS)   # Paraphrasing（軽量でOK）

MODEL_IDS = {
    "GEN":  OPENAI_MODEL_ID_GEN,
    "CLS":  OPENAI_MODEL_ID_CLS,
    "TQG":  OPENAI_MODEL_ID_TQG,
    "PARA": OPENAI_MODEL_ID_PARA,
}

# --- Answer structure judging (Step 4) ---
REQUIRE_VALID_EXAMPLE = True        # valid example が入るまで追い質問する
REQUIRE_BOUNDARY_OR_COUNTER = False # Trueにすると例 + (境界or反例)が揃うまで追う

SHOW_LEFT_PANEL = True
SHOW_RIGHT_PANEL = False  # ← 右を消したいなら False

EXIT_COMMANDS = {"exit", "q", "quit", "終了", "おわり", "終わり"}

# --- TeachingHelper: Chatに残す要約用ラベル ---
ANTI_LABELS = {
    "Commanding":    "Commanding（指示/修正に偏り）",
    "Spoon-feeding": "Spoon-feeding（答えの提示に偏り）",
    "Under-teaching":"Under-teaching（情報が薄い）",
    "Default":       "Default（良い流れ）",
}

HELPER_OPTION_LABELS = {
    "理解チェック質問（今の理解を言ってもらう）": "理解を説明してもらう",
    "一部を穴埋めにして考えさせる": "穴埋めで考えさせる",
    "『もし〜なら？』の思考実験を投げる": "条件を変えて考えさせる",

    "理由（why）を質問する": "理由を説明してもらう",
    "例や反例を求める": "例/反例で確認する",
    "境界条件（low/high, mid更新）を一緒に確認する": "境界条件を確認する",
}
HELPER_OPTION_HINTS = {
    "理解チェック質問（今の理解を言ってもらう）": "相手の理解を言語化させる質問に変更します。",
    "一部を穴埋めにして考えさせる": "答えを出し切らず、1ステップだけ相手に埋めさせます。",
    "『もし〜なら？』の思考実験を投げる": "条件を変えた場合の挙動を考えさせます。",

    # Commanding ←★これを追加
    "理由（why）を質問する": "修正指示ではなく、「なぜそうなるのか」をAlgoBoに説明させる質問に変えます。",
    "例や反例を求める": "具体的な入力例/失敗例を出させて、理解の穴を見つけやすくします。",
    "境界条件（low/high, mid更新）を一緒に確認する": "low/high/mid の更新や while 条件など、バグりやすい端のケースを確認する質問に変えます。",
}

# --- 論文寄せの追加設定 ---
HELPER_EVERY_N_MESSAGES = 6        # Teaching Helperは6メッセージごとに出す :contentReference[oaicite:2]{index=2}
QUESTION_LOOP_MAX_FOLLOWUPS = 8    # constructive loopの最大追い質問回数（運用で調整）

# --- 論文寄せ：constructive loop 判定/追い質問生成 ---
USE_QUALITY_FOR_LOOP_END = True          # 論文の Response Quality Classifier っぽく GOOD/BAD で終了判定
FOLLOWUP_USE_PARAPHRASE = True           # プロトコル(固定文)→文脈に合わせて言い換え
FOLLOWUP_AVOID_REPEAT_IN_LOOP = True     # 同一ループ内で同じプロトコルを避ける

# --- logging knobs (optional) ---
LOG_TIMING_TRACE = True
LOG_QUESTION_GEN_TRACE = True
LOG_CONTEXT_TEXT_IN_TRACE = True  # context_textをtraceに残す（重い/機微ならFalse）

# --- 知識状態の型定義 ---
class KnowledgeState(TypedDict):
    facts: List[str]
    code_implementation: List[str]

# --- OpenAI API 初期化 ---
def get_openai_client():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        return None

# --- LLM呼び出し関数 ---
def _to_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def call_gpt(client, system_prompt: str, user_prompt: str, temperature=0.0, max_tokens=1024, model_id: Optional[str] = None):
    if not client:
        return None

    st.session_state["_last_llm_meta"] = None

    model_id = model_id or OPENAI_MODEL_ID_GEN  # デフォルトは生成モデル

    params = {
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

        text = getattr(response, "output_text", "") or ""

        # ★追加：打ち切り理由・トークン使用量などを保存
        st.session_state["_last_llm_meta"] = {
            "model_id": model_id,
            "status": getattr(response, "status", None),
            "usage": _to_plain_data(getattr(response, "usage", None)),
        }

        return text

    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return None


def call_role(client, role: str, system_prompt: str, user_prompt: str,
              temperature=0.0, max_tokens=1024):
    model_id = MODEL_IDS.get(role, OPENAI_MODEL_ID_GEN)
    return call_gpt(
        client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model_id=model_id
    )


# --- ヘルパー関数 ---
def clamp_sentences_ja(text: str, max_sentences: int = 3, max_chars: int = 500) -> str:
    if not text:
        return ""
    parts = re.split(r'(?<=[。！？.!?])|\n+', text)
    parts = [p.strip() for p in parts if p.strip()]
    out = "".join(parts[:max_sentences]) if parts else text.strip()
    out = out.strip()

    if len(out) > max_chars:
        cut = out[:max_chars]
        # できるだけ「文末/句読点」まで戻す
        m = re.search(r'[。！？.!?](?!.*[。！？.!?])', cut)
        if m:
            cut = cut[:m.end()]
        out = cut.strip()
    return out

def _is_trivial_question(q: str) -> bool:
    q = (q or "").strip()
    # "?" "？？" だけ、または実質文字がほぼ無い
    if re.fullmatch(r"[？?]+", q):
        return True
    core = re.sub(r"[？?\s]+", "", q)
    return len(core) < 2


def normalize_question(text: str, max_chars: int = 240) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)

    qm = [m.start() for m in re.finditer(r"[？?]", t)]
    if qm:
        end = qm[-1] + 1

        seps = ["。", "！", "!", ".", "．"]
        start = 0
        best_start = 0
        for sep in seps:
            p = t.rfind(sep, 0, end - 1)
            if p != -1:
                best_start = max(best_start, p + 1)
        start = best_start

        candidate = t[start:end].strip()

        # ★ここが追加：candidateが "?" だけなら、文切り出しを撤回して全体を残す
        if _is_trivial_question(candidate):
            candidate = t[:end].strip()

        t = candidate
    else:
        if not t.endswith(("？", "?")):
            t += "？"

    if len(t) > max_chars:
        t = t[-max_chars:].lstrip("、。．.!！ ")
        if not t.endswith(("？", "?")):
            t += "？"

    if not t.endswith(("？", "?")):
        t += "？"

    # 最終保険：それでも "?" しかないなら固定の質問にフォールバック
    if _is_trivial_question(t):
        t = "どこが一番バグりやすいと思いますか？"

    return t


def _looks_like_question_sentence(q: str) -> bool:
    q = (q or "").strip()

    # まず trivial は弾く
    if _is_trivial_question(q):
        return False

    # 「どんな結果になるのはなぜ」のように、what/how と why が混ざった不自然な質問を弾く
    awkward_patterns = [
        r"どんな.+(のは|なのは)なぜ",
        r"どのような.+(のは|なのは)なぜ",
        r"何.+(のは|なのは)なぜ",
    ]
    if any(re.search(p, q) for p in awkward_patterns):
        return False

    # 1文ルール（句点が多い/改行がある）は弾く
    if "\n" in q:
        return False
    if q.count("。") >= 1:
        # 例外的に「…ですか？」の直前に1個だけなら許す、なども可
        return False

    # 疑問の形っぽい語尾がない長文は弾く（保険）
    tail = q[-8:]
    if ("ですか" not in q) and ("ますか" not in q) and ("かな" not in q) and (tail.endswith(("？","?")) is False):
        return False

    # 10文字未満は弱すぎる
    core = re.sub(r"[？?\s]+", "", q)
    if len(core) < 10:
        return False

    return q.endswith(("？","?"))


def generate_question_safe(client, context, past_qs, qtype="WHY"):
    trace = {
        "qtype": qtype,
        "past_qs_n": len(past_qs or []),
        "raw": "",
        "normalized1": "",
        "paraphrase_raw": "",
        "normalized2": "",
        "validation_passed": False,
        "fallback_level": 0,  # 0=通常OK, 1=強指示で再生成, 2=最終固定文
    }
    if LOG_CONTEXT_TEXT_IN_TRACE:
        trace["context_text"] = context

    # 1) 通常生成
    raw_q = generate_question(client, context, past_qs, qtype=qtype) or ""
    q = normalize_question(raw_q)
    trace["raw"] = raw_q
    trace["normalized1"] = q

    # 2) paraphrase（任意）
    q2 = paraphrase(client, q, context)  # paraphrase内で normalize_question 済み
    trace["paraphrase_raw"] = st.session_state.get("_last_paraphrase_raw", "")
    q2 = normalize_question(q2)
    trace["normalized2"] = q2

    # 3) 検査 → ダメなら “より強い指示” で再生成
    if not _looks_like_question_sentence(q2):
        trace["fallback_level"] = 1
        regen = call_role(
            client,
            role="TQG",
            system_prompt="""
あなたはAlgoBo。次を厳守して質問文を1つだけ出力。
- 12〜60文字程度
- 1文、改行禁止、説明禁止
- 「？」で終了
- 初学者に自然な言葉にする
- 「不変条件」「単調性」のような専門語は禁止
- 「どんな結果になるのはなぜですか」のように「どんな」と「なぜ」を混ぜた不自然な文は禁止
- 聞き方は「どんな結果になりますか？」または「なぜ間違えますか？」のどちらか一方にする
- 具体語を1つ含める（low/high/mid/while/反例/具体例/計算量/ソート のどれか）
""",
            user_prompt=f"[CONTEXT]\n{context}\n\n質問文を1つだけ出力:",
            temperature=0.7,
            max_tokens=128,
        ) or ""
        q2 = normalize_question(regen)
        trace["raw_regen"] = regen
        trace["normalized2"] = q2

    # 4) それでもダメなら最終フォールバック
    if not _looks_like_question_sentence(q2):
        trace["fallback_level"] = 2
        q2 = "low/high を更新するときに無限ループを避ける条件は何ですか？"
        trace["normalized2"] = q2

    trace["validation_passed"] = _looks_like_question_sentence(q2)

    # ★追加：今回の質問生成traceを保存（turnsに入れる用）
    st.session_state["_last_question_gen_trace"] = trace

    return q2


def first_sentence_ja(text: str, fallback_chars: int = 60) -> str:
    """
    日本語の先頭1文を取り出す。
    - 最初の 。！？!? まで
    - それが無ければ fallback_chars で切る
    """
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)

    m = re.search(r"[。！？!?]", t)
    if m:
        return t[:m.end()].strip()

    # 句点が無い場合の保険
    if len(t) > fallback_chars:
        return (t[:fallback_chars] + "…").strip()
    return t


def is_tutor_question(classification: str, text: str) -> bool:
    """
    Tutorが質問している時は、AlgoBo側は質問生成に入らず回答モードを優先するためのガード。
    誤爆しやすい「? を含むだけ」を避け、末尾の ?/？ を基本に判定する。
    """
    t = (text or "").strip()

    # 1) 分類で質問系（ただし分類はブレるので、必要なら末尾?と併用でもOK）
    q_like = {"Checking", "Thought-provoking", "Challenge-finding"}
    if classification in q_like:
        return True

    # 2) 三項演算子っぽい `?:` を含むだけなら質問扱いしない（末尾?は別）
    #    例: (x>0) ? a : b
    if ("?:" in t) or re.search(r"\?\s*:", t):
        # 末尾が ? ならそれは質問として扱う
        return bool(re.search(r"[？?]\s*$", t))

    # 3) 基本：末尾が ?/？
    if re.search(r"[？?]\s*$", t):
        return True

    # 4) 保険：日本語の疑問っぽい終わり（？が無い質問文対策）
    #    例: 「どうしてソートが必要なのか教えてください」
    if re.search(r"(ですか|ますか|でしょうか|教えて(ください)?|説明して(ください)?|なぜ|どうして|どうやって|何が|どれが|いつ|どこ|誰)\s*$", t):
        return True

    return False


def looks_like_concrete_example(answer: str, question: str = "") -> bool:
    a = (answer or "")
    q = (question or "")
    q_low = q.lower()

    # --- 配列の検出（Python + C）---
    # Python: arr=[1,2,3]
    has_py_arr = bool(re.search(r"\b(arr|a)\s*=\s*\[[^\]]+\]", a, re.IGNORECASE))
    # C: int arr[] = {1,2,3}; / {1,2,3} が含まれる（宣言がなくてもOKにする）
    has_c_arr_decl = bool(re.search(r"\b(?:int|long|short|char)\s+(arr|a)\s*\[\s*\]\s*=\s*\{[^}]+\}", a, re.IGNORECASE))
    has_c_init = bool(re.search(r"\{[^}]*\d[^}]*\}", a))  # {}の中に数字がある
    has_index_assign = bool(re.search(r"\b(arr|a)\s*\[\s*\d+\s*\]\s*=", a, re.IGNORECASE))
    # 文章中に [] が出るだけでも「配列っぽい」扱い（弱いが保険）
    has_brackets = ("[" in a and "]" in a)

    has_arr = has_py_arr or has_c_arr_decl or has_index_assign or has_c_init or has_brackets

    # --- target/x/key の検出（Python + C + 日本語表現）---
    # 例: target=7 / x=7 / key=7 / target は 7
    has_target_assign = bool(re.search(r"\b(target|x|key)\s*(=|は)\s*-?\d+", a, re.IGNORECASE))
    # C: int target = 7;
    has_c_target_decl = bool(re.search(r"\b(?:int|long|short|char)\s+(target|x|key)\s*=\s*-?\d+", a, re.IGNORECASE))
    # 日本語: 探す値は7 / 検索値=7 / キーは7
    has_jp_target = bool(re.search(r"(探す値|検索値|キー)\s*(は|=)\s*-?\d+", a))

    has_target = has_target_assign or has_c_target_decl or has_jp_target

    # --- low/high/mid のトレース検出（Python + C）---
    # 例: low=0 high=4 mid=2 / low:0, high:4, mid:2
    has_lhm_nums = bool(re.search(r"\b(low|high|mid)\b\s*[:=]?\s*-?\d+", a, re.IGNORECASE))

    # Python: mid = (low+high)//2 や //2
    has_py_mid = ("//2" in a) or bool(re.search(r"\(\s*low\s*\+\s*high\s*\)\s*//\s*2", a, re.IGNORECASE))

    # C: mid = (low + high) / 2; など
    has_c_mid = bool(re.search(r"\(\s*low\s*\+\s*high\s*\)\s*/\s*2", a, re.IGNORECASE)) or \
                bool(re.search(r"\bmid\b\s*=\s*\(\s*low\s*\+\s*high\s*\)\s*/\s*2", a, re.IGNORECASE))

    has_mid_trace = has_lhm_nums or has_py_mid or has_c_mid

    # --- 質問が mid/中央/偶数長 などなら、target必須にせず「配列＋トレース」を重視 ---
    about_mid = (
        ("mid" in q_low) or ("中央" in q) or ("真ん中" in q) or ("偶数" in q) or ("low" in q_low) or ("high" in q_low)
    )

    if about_mid:
        return has_arr and has_mid_trace

    # 通常は「配列 + target」が最低条件
    return has_arr and has_target




def _ensure_str(x, label=""):
    if isinstance(x, str):
        return x
    try:
        st.session_state.setdefault("logs", [])
        st.session_state.logs.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "non_string_content_detected",
            "label": label,
            "value_type": str(type(x)),
            "repr": repr(x)[:200],
        })
    except:
        pass
    return str(x)



def append_helper_to_chat(helper_state: dict, turn_num: int, antipattern: str, blocked: bool):
    """Teaching Helper を会話(messages)に要約形式で1回だけ残す（rerunで重複しない）"""
    if not helper_state:
        return

    kind = _ensure_str(helper_state.get("kind", "green"), label="helper.kind")
    body = _ensure_str(helper_state.get("body", ""), label="helper.body")

    headline = first_sentence_ja(body)
    anti_label = ANTI_LABELS.get(antipattern, antipattern or "Unknown")

    # 初期状態（choiceは未選択）
    choice_label = "-"
    status = "ブロック（未送信）" if blocked else "ブロックなし"

    # ★重複判定キー（本文ではなくturn+kind+antipatternで）
    key = (turn_num, kind, antipattern)
    if key in st.session_state.helper_chat_keys:
        return
    st.session_state.helper_chat_keys.add(key)

    emoji = "🟥" if kind == "red" else "🟩"

    lines = [
        f"{emoji} Teaching Helper: {anti_label}",
        headline,
        f"選択した改善方針: {choice_label}",
        f"状態: {status}",
    ]

    st.session_state.messages.append({
        "role": "assistant",
        "content": "\n".join(lines),
        "meta": {
            "type": "helper",
            "turn": turn_num,
            "kind": kind,
            "antipattern": antipattern,
            "headline": headline,
            "choice": choice_label,
            "status": status,
        }
    })


def update_helper_chat_summary(turn_num: int, choice: Optional[str], status: str):
    """既にmessagesに入っている helper 要約を、choice/status だけ更新する"""
    if not turn_num:
        return

    choice_label = HELPER_OPTION_LABELS.get(choice, choice) if choice else "-"

    for i in range(len(st.session_state.messages) - 1, -1, -1):
        m = st.session_state.messages[i]
        meta = m.get("meta", {})
        if meta.get("type") == "helper" and meta.get("turn") == turn_num:
            kind = meta.get("kind", "green")
            antipattern = meta.get("antipattern", "Unknown")
            headline = meta.get("headline", "")

            anti_label = ANTI_LABELS.get(antipattern, antipattern)
            emoji = "🟥" if kind == "red" else "🟩"

            lines = [
                f"{emoji} Teaching Helper: {anti_label}",
                headline,
                f"選択した改善方針: {choice_label}",
                f"状態: {status}",
            ]
            m["content"] = "\n".join(lines)

            # meta も更新（後でログ解析しやすい）
            meta["choice"] = choice_label
            meta["status"] = status
            m["meta"] = meta
            break



def log_helper_shown(turn_num: int, helper_state: dict, blocked: bool,
                     antipattern: str, ap_stats: Optional[dict],
                     original_prompt: Optional[str], original_classification: Optional[str]):
    if not helper_state:
        return

    kind = _ensure_str(helper_state.get("kind", ""), label="helper.log.kind")
    body = _ensure_str(helper_state.get("body", ""), label="helper.log.body_key")
    key = (turn_num, kind, body, bool(blocked))

    if key in st.session_state.helper_log_keys:
        return
    st.session_state.helper_log_keys.add(key)

    st.session_state.logs.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "type": "helper_shown",
        "turn_num": turn_num,
        "antipattern": antipattern,
        "blocked": bool(blocked),
        "helper": {
            "kind": helper_state.get("kind"),
            "title": helper_state.get("title"),
            "body": _ensure_str(helper_state.get("body", ""), label="helper.log.body"),
            "options": helper_state.get("options", []),
            "debug": _ensure_str(helper_state.get("debug", ""), label="helper.log.debug") if helper_state.get("debug") else "",
        },
        # ブロック時に「未送信の元案」も確定保存できる
        "original_prompt": original_prompt,
        "original_classification": original_classification,
        "antipattern_stats": ap_stats,
    })


def build_llm_context(last_k: int = 3) -> str:
    """LLMに渡す会話文脈：Teaching Helper(meta)は除外"""
    msgs = [m for m in st.session_state.messages if m.get("meta", {}).get("type") != "helper"]
    return "\n".join([m["content"] for m in msgs[-last_k:]])


def build_tutor_only_context(last_k: int = 6) -> str:
    msgs = [m for m in st.session_state.messages if m.get("role") == "user"]
    return "\n".join([m["content"] for m in msgs[-last_k:]])


def _strip_md(s: str) -> str:
    s = s or ""
    # 太字/斜体などの * _ `
    s = re.sub(r"[*_`]+", "", s)
    # 箇条書きの先頭
    s = re.sub(r"^\s*[-•]\s*", "", s, flags=re.MULTILINE)
    # 見出し #
    s = re.sub(r"^\s*#+\s*", "", s, flags=re.MULTILINE)
    return s


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _strip_md(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _norm_code(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _strip_md(s)
    s = re.sub(r"\s+", "", s).strip().lower()
    return s


def extract_json_block(text: str) -> dict:
    if not text:
        raise ValueError("empty")
    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("no json object found")
    return json.loads(text[l:r+1])


# --- ★ここから詳細プロンプト版のロジック関数 ---

def op_extract(client, conversation):
    # 【詳細版】抽出プロンプト
    system_prompt = f"""
    あなたは、会話の内容を分析し、AlgoBoの知識状態に追加すべき新しい知識（概念やコード）を正確に抽出する専門家です。

    【厳守事項】
    1. 抽出された知識は、**必ず**リスト形式のJSONオブジェクトとして提供してください。
    2. JSON以外に、**説明やコメント、前置き、後書き（「以下に抽出しました」など）を一切追加しないでください。**純粋なJSON文字列のみを出力してください。
    3. 知識がない場合は、空のJSONオブジェクト（例: {{"knowledge": []}}）を返してください。
    4．CONVERSATIONに登場する文言をそのまま抽出してください（言い換え禁止）。
    5．CONVERSATIONに無い知識を推測して追加するのは禁止です。

    [知識の形式]
    {{
        "knowledge": [
            "抽出された自然言語での新しい事実",
            "抽出されたコードスニペット（必ずプログラミング言語名から開始）"
        ]
    }}
    """
    user_prompt = f"以下のCONVERSATIONから、AlgoBoにとって新しい、重要な知識を抽出してください。\n\nCONVERSATION:\n{conversation}"

    res = call_role(
        client,
        role="GEN",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=1024,
    )
    # ---- JSONをなるべく安全に抜く（最初の { から最後の } まで）----
    m = re.search(r"\{.*\}", res, re.DOTALL)  # 非貪欲だと途中で切れやすい
    cleaned = (m.group(0) if m else res).strip()

    try:
        obj = json.loads(cleaned)
        raw = obj.get("knowledge", [])
        if not isinstance(raw, list):
            return []
    except Exception:
        return []

    # ---- 「会話に存在するものだけ採用」：厳しすぎないフィルタ ----
    conversation_text = conversation or ""
    nc_text = _norm_text(conversation_text)
    nc_code = _norm_code(conversation_text)

    sentences = re.split(r'(?<=[。！？.!?])\s*|\n+', conversation_text)
    sentences = [x for x in sentences if x.strip()]

    def best_ratio(item_norm: str) -> float:
        best = 0.0
        for s in sentences:
            sn = _norm_text(s)
            if not sn:
                continue
            r = difflib.SequenceMatcher(None, item_norm, sn).ratio()
            if r > best:
                best = r
        return best

    filtered = []
    for item in raw:
        item = (item or "").strip()
        if not item:
            continue

        ni_text = _norm_text(item)
        ni_code = _norm_code(item)

        ok = (ni_text and ni_text in nc_text) or (ni_code and ni_code in nc_code)

        if not ok and ni_text:
            # だいたい同じ文ならOK（0.85は無難。拾わなすぎるなら0.82〜）
            if best_ratio(ni_text) >= 0.85:
                ok = True

        if ok:
            filtered.append(item)

    return filtered


def op_update(client, current_state, new_knowledge):
    if not new_knowledge: return current_state

    new_knowledge_str = "\n".join([f"- {item}" for item in new_knowledge])
    current_knowledge_str = json.dumps(current_state, ensure_ascii=False, indent=2)

    # 【詳細版】更新プロンプト
    system_prompt = """
    あなたは知識統合の専門家です。
    [KNOWLEDGE]（現在の知識状態）と [NEW KNOWLEDGE]（新しい知識）が与えられます。
    以下のルールに従って知識を更新し、JSON形式で出力してください。

    【更新ルール】
    1. **統合 (Merge):** [KNOWLEDGE] 内に [NEW KNOWLEDGE] と関連する項目がある場合、それらを1つの簡潔な事実にまとめてください。重複してはいけません。
    2. **追加 (Add):** [NEW KNOWLEDGE] が全く新しい情報であれば、[KNOWLEDGE] に追加してください。
    3. **修正 (Correct):** 新しい知識が既存の知識と矛盾する場合、新しい知識を優先して内容を修正してください。
    4. **分類:**
       - 自然言語の説明は "facts" リストに入れてください。
       - プログラミングコード（C言語コードなど）は "code_implementation" リストに入れてください。
    5. **簡潔性:** 知識状態全体を可能な限り短く、簡潔に保ってください。

    【出力フォーマット】
    JSONオブジェクトのみを出力してください。Markdownのコードブロック（```json）や説明文は不要です。
    {
      "facts": ["..."],
      "code_implementation": ["..."]
    }
    """
    user_prompt = f"[KNOWLEDGE]\n{current_knowledge_str}\n\n[NEW KNOWLEDGE]\n{new_knowledge_str}\n\n更新された知識状態(JSON)を出力してください:"

    res = call_role(
        client,
        role="GEN",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=1024,
    )
    try:
        updated = extract_json_block(res)
        return {
            "facts": updated.get("facts", []),
            "code_implementation": updated.get("code_implementation", [])
        }
    except:
        return current_state


def op_retrieve(client, current_state, context):
    # 知識をインデックス付き文字列に変換
    knowledge_str = ""
    for idx, fact in enumerate(current_state["facts"]): knowledge_str += f"FACT_{idx}: {fact}\n"
    for idx, code in enumerate(current_state["code_implementation"]): knowledge_str += f"CODE_{idx}: {code}\n"

    # 【詳細版】検索プロンプト
    system_prompt = f"""
    あなたは、AlgoBoの現在の知識状態（KNOWLEDGE）と会話の文脈（CONVERSATION CONTEXT）を分析し、応答生成に必要となる**最も関連性の高い知識のインデックス**を特定する専門家です。
    インデックスはJSON形式で提供してください。インデックスは最大3つまで含めることができます。

    [インデックス形式]
    {{
        "relevant_knowledge_indices": ["FACT_0", "CODE_1", ...]
    }}

    もし関連する知識がない場合は、空のJSONオブジェクト（例: {{"relevant_knowledge_indices": []}}）を返してください。
    """
    user_prompt = f"[KNOWLEDGE]\n{knowledge_str}\n\n[CONVERSATION CONTEXT]\n{context}\n\n上記に基づいて、AlgoBoの応答に必要な知識のインデックスを抽出してください。"

    res = call_role(
        client,
        role="GEN",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=256,
    )
    retrieved_content = []
    try:
        match = re.search(r'\{.*?\}', res, re.DOTALL)
        cleaned = match.group(0) if match else res.strip()
        relevant_indices = json.loads(cleaned).get("relevant_knowledge_indices", [])

        for index_str in relevant_indices:
            try:
                if index_str.startswith("FACT_"):
                    idx = int(index_str.split("_")[1])
                    if idx < len(current_state["facts"]): retrieved_content.append(current_state["facts"][idx])
                elif index_str.startswith("CODE_"):
                    idx = int(index_str.split("_")[1])
                    if idx < len(current_state["code_implementation"]): retrieved_content.append(current_state["code_implementation"][idx])
            except: continue
    except: pass
    return retrieved_content


def op_check_consistency(client, current_state, new_knowledge):
    if not new_knowledge:
        return {
            "verdict": "NO_NEW_KNOWLEDGE",
            "summary": "",
            "conflict_with": "",
        }

    current_items = (current_state.get("facts", []) or []) + (current_state.get("code_implementation", []) or [])
    if not current_items:
        return {
            "verdict": "NO_PRIOR_KNOWLEDGE",
            "summary": "既存知識がまだないため、この内容を最初の知識として確認します。",
            "conflict_with": "",
        }

    current_text = "\n".join(current_items)
    new_text = "\n".join(new_knowledge)
    current_norm = _norm_text(current_text)
    new_norm = _norm_text(new_text)

    # 二分探索ドメインで重要な典型ケースは、分類モデルに任せず先に固定判定する。
    says_unsorted_always_ok = (
        ("ソートされていなくても" in new_norm or "並んでいなくても" in new_norm or "整列されていなくても" in new_norm)
        and ("常に" in new_norm or "いつでも" in new_norm or "必ず" in new_norm)
        and ("正しく" in new_norm or "探せ" in new_norm or "探索" in new_norm)
    )
    prior_depends_on_order = (
        ("並んでいる" in current_norm or "ソート" in current_norm or "整列" in current_norm)
        and ("左" in current_norm or "右" in current_norm or "半分" in current_norm or "真ん中" in current_norm or "中央" in current_norm)
    )
    if says_unsorted_always_ok and prior_depends_on_order:
        return {
            "verdict": "CONFLICT",
            "summary": "未ソートでも常に正しく探せるという説明は、並んだデータで左右を判断するという既存知識と矛盾します。",
            "conflict_with": current_items[0],
        }

    says_sort_needed_for_direction = (
        ("ソートが必要" in new_norm or "並んでいる必要" in new_norm or "整列が必要" in new_norm)
        and ("左" in new_norm or "右" in new_norm)
        and ("判断" in new_norm or "決め" in new_norm or "行く" in new_norm)
    )
    if says_sort_needed_for_direction and prior_depends_on_order:
        return {
            "verdict": "CONSISTENT",
            "summary": "ソートが左右判断に必要という補足は、並んだデータで範囲を半分に減らす既存知識と両立します。",
            "conflict_with": "",
        }

    current_knowledge_str = json.dumps(current_state, ensure_ascii=False, indent=2)
    new_knowledge_str = "\n".join([f"- {item}" for item in new_knowledge])

    system_prompt = """
あなたはAlgoBoの知識状態を点検する評価者です。
[CURRENT KNOWLEDGE] と [LATEST KNOWLEDGE] を比較し、明確な矛盾があるか判定してください。

判定ルール:
- 言い換え、詳細化、補足、抽象度の違いは CONSISTENT
- 片方にしか書かれていない情報は、反対内容でない限り CONSISTENT
- 明確に両立しない内容だけ CONFLICT
- 判断に迷う場合は UNCLEAR

出力は必ずJSONのみ:
{
  "verdict": "CONSISTENT" | "CONFLICT" | "UNCLEAR",
  "summary": "短い判定理由",
  "conflict_with": "矛盾する既存知識の短い抜粋。なければ空文字"
}
"""
    user_prompt = f"[CURRENT KNOWLEDGE]\n{current_knowledge_str}\n\n[LATEST KNOWLEDGE]\n{new_knowledge_str}\n\n矛盾判定:"

    res = call_role(
        client,
        role="CLS",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=256,
    ) or ""

    try:
        obj = extract_json_block(res)
        verdict = str(obj.get("verdict", "UNCLEAR")).upper()
        if verdict not in {"CONSISTENT", "CONFLICT", "UNCLEAR"}:
            verdict = "UNCLEAR"
        return {
            "verdict": verdict,
            "summary": str(obj.get("summary", "")),
            "conflict_with": str(obj.get("conflict_with", "")),
        }
    except Exception:
        return {
            "verdict": "UNCLEAR",
            "summary": "矛盾判定のJSON解析に失敗しました。",
            "conflict_with": "",
        }


def op_compose(client, latest_knowledge, context, consistency_check=None):
    # 【詳細版】応答生成プロンプト
    persona = "あなたはAlgoBoという、プログラミングを学ぶ1年目の学生です。C言語の基本的なシンタックスは知っていますが、二分探索のロジックで苦労しています。親しみやすく、質問に熱心に応答しますが、知らないことには素直に助けを求めます。回答は全て日本語で行ってください。"

    if not latest_knowledge:
        return "二分探索についてもっと詳しく説明してもらえませんか？"

    latest_knowledge_str = "\n".join(latest_knowledge)
    consistency_check = consistency_check or {"verdict": "UNCLEAR", "summary": "", "conflict_with": ""}
    consistency_str = json.dumps(consistency_check, ensure_ascii=False, indent=2)
    system_prompt = f"""
    {persona}

    あなたの応答は、以下の制約を厳守してください:
    1. 応答は**最大2文**までで、簡潔にしてください。
    2. 基本は1文だけで、**[LATEST KNOWLEDGE]**に書かれた直前のTutor発話内容だけを確認する口調にしてください。
    3. [LATEST KNOWLEDGE]にない定義、手順、理由、例、補足知識を追加してはいけません。
    4. [CONVERSATION CONTEXT]は文脈把握のためだけに使い、新しい知識の根拠にしてはいけません。
    5. Tutorに教えてもらった内容を確認する口調にしてください（例: 「〜ということですね」「〜という理解で合っていますか？」）。
    6. 「〜なんだよ」「〜です！」のように先生としてまとめ直す口調は禁止です。
    7. **[CONSISTENCY CHECK]** の verdict が CONFLICT の場合だけ、2文目で既存知識と違って見える点を短く述べ、どちらで理解すべきか質問してください。
    8. verdict が CONSISTENT / NO_PRIOR_KNOWLEDGE / UNCLEAR の場合は、矛盾確認について触れないでください。「矛盾していません」「矛盾はなさそうです」「既存知識がまだない」なども言わないでください。
    9. 知識が不足している場合は、「二分探索についてもっと詳しく説明してもらえませんか？」といった**助けを求める形式**で応答してください。
    10. 親しみやすい学生のような口調を維持してください。
    11. 回答は**全て日本語**で行ってください。
    """
    user_prompt = f"[CONVERSATION CONTEXT]\n{context}\n\n[LATEST KNOWLEDGE]\n{latest_knowledge_str}\n\n[CONSISTENCY CHECK]\n{consistency_str}\n\nAlgoBoとして、直前発話の確認を短く応答してください。矛盾がある場合だけ、どちらで理解すべきかも確認してください。"

    res = call_role(
        client,
        role="GEN",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=512,
    )
    if not res or (not latest_knowledge and "分かりません" not in res): return "二分探索についてもっと詳しく説明してもらえませんか？"
    return clamp_sentences_ja(res, 2)

def generate_question(client, context, past_qs, qtype: str = "WHY"):
    past_q_str = "\n".join([f"- {q}" for q in past_qs])

    if qtype.upper() == "HOW":
        q_instruction = "現在の文脈に基づいて、Tutorに『どうやって？』『具体的に？』を問う質問を生成してください（手順、例、境界条件、失敗ケースなど）。"
    else:
        q_instruction = "現在の文脈に基づいて、Tutorに『なぜ？』の理由・根拠を問う質問を生成してください。"

    system_prompt = f"""
あなたはAlgoBoという、プログラミングを学ぶ熱心な学生です。Tutor（先生）の知識を深めるために、賢く、考えさせる質問をします。質問は全て日本語で行ってください。

【質問生成ルール】
1. 質問は最大1文
2. {q_instruction}
3. 下の【過去の質問】と内容が被る質問は禁止（言い換えも禁止）
4. 初学者に自然な言葉で聞く。「不変条件」「単調性」のような専門語は禁止
5. 違う観点（例：具体例、計算量、境界条件、失敗ケース、重複要素など）を狙う
6. 「どんな結果になるのはなぜですか」のように「どんな」と「なぜ」を混ぜた不自然な文は禁止
7. 聞き方は「どんな結果になりますか？」または「なぜ間違えますか？」のどちらか一方にする
8. 出力は「？（または?）」で終わる質問文だけにしてください。説明文は禁止。改行も禁止。

【過去の質問】
{past_q_str}
"""
    user_prompt = f"[現在の会話文脈]\n{context}\n\nこの文脈に基づいて、思考を促す質問を生成してください。"

    res = call_role(
        client,
        role="TQG",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=256
    )
    if not res:
        return "二分探索についてもっと詳しく説明してもらえませんか？"
    return normalize_question(res)


def paraphrase(client, raw_msg, context):
    # 【詳細版】言い換えプロンプト
    system_prompt = """
    あなたはプログラミングを学ぶ学生「AlgoBo」です。
    [RAW QUESTION] を、[CONVERSATION] の流れに自然に続く「質問文1つ」にしてください。

    【厳守】
    - 出力は質問文のみ（前置き・相槌・説明・結論は禁止）
    - 改行禁止
    - 最後は必ず「？」で終える
    - 1つの質問だけ（複数の？は禁止）
    """
    user_prompt = f"[CONVERSATION]\n{context}\n\n[RAW QUESTION]\n{raw_msg}\n\nAlgoBoとしての発言:"

    raw = call_role(
        client,
        role="PARA",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=256
    ) or ""

    # ★追加：paraphraseの生出力を保存（ログ用）
    st.session_state["_last_paraphrase_raw"] = raw

    return normalize_question(raw if raw else raw_msg)


def classify_message(client, msg):
    # 【詳細版】分類プロンプト
    system_prompt = f"""
    あなたは、LBT（Learning by Teaching）対話の専門家です。Tutor（先生）の最新のメッセージを分析し、以下のタクソノミー（Table 1より）の**Sub Category**一つに分類してください。

    [タクソノミー（抜粋）]
    1. Instruction (指示): Fixing (コード/知識の修正指示), Commanding (シンプルな行動指示), Encouraging (感情的な励まし)
    2. Prompting (質問/要求): Challenge-finding (Tuteeの苦労点を問う), Hinting (代替案の検討を促す), Checking (理解度の確認), Thought-provoking (深い考察/ elaborationを促す)
    3. Statement (発言): Comprehension (知識の伝達/説明), Elaboration (詳細な説明/例), Sense-making (エラー修正の気づき), Accepting/Reject (同意/不同意), Feedback (評価)

    【厳守事項】
    1. 分類結果は、**最も適切な Sub Category の名前のみ**を返してください。
    2. JSONや追加の説明は不要です。
    3. 全て英語の Sub Category 名で返してください（例: Fixing, Comprehension, Thought-provoking）。
    """
    user_prompt = f"以下のTutorメッセージを分類してください:\n\"{msg}\""
    res = call_role(
        client,
        role="CLS",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=128
    )
    if not res:
        return "Unknown"
    return res.strip().replace('"', '').replace("'", '').split('\n')[0]


def judge_answer_quality(client, question: str, answer: str, context: str) -> str:
    """
    論文は学習した Response Quality Classifier を使う :contentReference[oaicite:12]{index=12}
    ここでは近似としてLLMで GOOD/BAD を判定する
    """
    system_prompt = """
あなたは教育対話の評価者です。
QUESTIONに対するANSWERが「深い回答」か判定してください。
深い回答の条件（例）:
- なぜ/根拠がある
- 具体例 or 反例 or 境界条件がある
- 手順が曖昧でなく、誤解が減る

出力は GOOD または BAD のどちらか1語のみ。
"""
    user_prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{question}\n\n[ANSWER]\n{answer}\n"
    res = call_role(
        client,
        role="CLS",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=128,
    ) or ""
    res = res.strip().upper()
    return "GOOD" if res.startswith("GOOD") else "BAD"


def summarize_answer_as_algobo(client, question: str, answer: str) -> str:
    system_prompt = """
あなたはAlgoBoです。Tutorの回答を聞いて理解した内容を最大2文で要約してください。
口調は親しみやすく、日本語で、最後は「！」を入れてください。
"""
    user_prompt = f"[QUESTION]\n{question}\n\n[ANSWER]\n{answer}\n\n要約:"
    res = call_role(
        client,
        role="GEN",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=256,
    ) or "なるほど、理由と具体例が大事なんですね！"
    return clamp_sentences_ja(res, 2)

def judge_answer_structure(client, question: str, answer: str, context: str) -> dict:
    """
    Tutor回答が、例/反例/境界条件を含むかを構造化(JSON)で判定する。
    返り値例:
    {
      "has_example": true,
      "has_counterexample": false,
      "has_boundary": true,
      "example_excerpt": "...",
      "counterexample_excerpt": "",
      "boundary_excerpt": "...",
      "verdict": "GOOD"
    }
    """
    system_prompt = """
あなたは教育対話の評価者です。
ANSWER が以下を含むかを判定し、必ず JSON だけを返してください（説明文禁止）。

[定義]
- example: 具体的な入力例（配列・targetなど具体値）と結果/途中経過が含まれる（例: arr=[1,3,5], target=3 → 1、またはmidがどう動く等）
- counterexample: うまくいかない例・誤る例（例: ソートされていない配列で破綻する等）
- boundary: 境界条件/端のケース（空配列, 1要素, 見つからない, 重複, low/high更新, low<=high vs low<high など）

[出力JSONフォーマット]
{
  "has_example": true/false,
  "has_counterexample": true/false,
  "has_boundary": true/false,
  "example_excerpt": "ANSWERから根拠の短い抜粋（なければ空）",
  "counterexample_excerpt": "同上",
  "boundary_excerpt": "同上"
}

制約:
- 必ずJSONのみ
- 抜粋は短く（1フレーズ程度）
"""

    user_prompt = f"""[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
{answer}

上の定義に従ってJSONで判定してください。"""

    res = call_role(
        client,
        role="CLS",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=256,
    ) or ""

    # JSON抽出
    data = {
        "has_example": False,
        "has_counterexample": False,
        "has_boundary": False,
        "example_excerpt": "",
        "counterexample_excerpt": "",
        "boundary_excerpt": "",
    }
    try:
        obj = extract_json_block(res)
        for k in data.keys():
            if k in obj:
                data[k] = obj[k]
    except:
        pass


    # ★保険：LLMが has_example:true と言っても「具体値が無い」例っぽい説明を弾く
    if data.get("has_example") and not looks_like_concrete_example(answer, question):
        data["has_example"] = False
        data["example_excerpt"] = ""

    # verdict（ここが「valid exampleが入るまで」の中核）
    has_ex = bool(data.get("has_example"))
    has_ce = bool(data.get("has_counterexample"))
    has_bd = bool(data.get("has_boundary"))

    if REQUIRE_VALID_EXAMPLE and not has_ex:
        verdict = "BAD"
    elif REQUIRE_BOUNDARY_OR_COUNTER and not (has_bd or has_ce):
        verdict = "BAD"
    else:
        verdict = "GOOD"

    data["verdict"] = verdict
    return data


# --- Constructive Tutee Inquiry Protocol (論文寄せ: 固定プロトコル群) ---
FOLLOWUP_PROTOCOLS = {
    # 具体例（トレース）要求
    "example_general": [
        ("ex_minimal", "いちばん小さい具体例（arrとtarget）を1つだけ作って、mid/low/highの更新を1〜2回だけ追って見せてもらえますか？"),
        ("ex_not_found", "見つからないケースの具体例（arrとtarget）を1つ出して、low>highになるまでの流れを短く追って説明できますか？"),
        ("ex_duplicates", "重複がある具体例（例: arr=[1,2,2,2,3]）で、どのindexを返す設計かも含めて説明できますか？"),
    ],

    # mid/偶数長の「中央」問題向け
    "mid_choice": [
        ("mid_even", "要素数が偶数の配列（例: [1,3,5,7]）で、midを左寄り/右寄りに取ったときの違いを1〜2手だけ追って見せてもらえますか？"),
        ("mid_progress", "midの取り方を変えても必ず区間が縮むために、low/high更新で守るべき条件を自分の言葉でまとめてもらえますか？"),
    ],

    # 計算量・log系
    "complexity": [
        ("cx_double", "Nが2倍になったとき比較回数がどう増えるか、N=8→16みたいな具体例で言えますか？"),
        ("cx_recurrence", "『毎回半分に減る』を式で書くとどうなって、そこからlogが出る流れを説明できますか？"),
        ("cx_1third", "もし毎回『半分』じゃなく『1/3』だけ減るなら、回数はどう変わるイメージですか？"),
    ],

    # 反例（ソートされてない等）
    "counterexample": [
        ("ce_unsorted", "ソートされていない配列で二分探索が破綻する反例を1つ出して、どの比較で判断がズレるか説明できますか？"),
        ("ce_wrong_update", "low=mid や high=mid をやってしまうと無限ループになる反例を1つ作れますか？"),
    ],

    # 境界条件（端ケース）
    "boundary": [
        ("bd_empty", "空配列・1要素・2要素のとき、while条件と更新がどうなるのが安全ですか？"),
        ("bd_loweqhigh", "探索範囲が1要素（low==high）になったときの処理を、returnの条件込みで説明できますか？"),
        ("bd_condition", "while low<high と low<=high の違いは、どんなケースで結果に影響しますか？"),
    ],

    # “根拠” を初学者向けの言葉で言わせる（例がなくても成立しやすい）
    "invariant_reason": [
        ("inv_invariant", "二分探索で『こっち側には答えがない』と言える理由を1文で説明できますか？"),
        ("inv_monotonic", "ソートされていないと左右のどちらを探すか決めにくい理由を、自分の言葉で説明できますか？"),
    ],
}


def _topic_flags(q: str):
    q = q or ""
    q_low = q.lower()
    about_mid = ("mid" in q_low) or ("中央" in q) or ("真ん中" in q) or ("偶数" in q) or ("2の冪" in q)
    about_complexity = ("log" in q_low) or ("計算量" in q) or ("比較回数" in q) or ("対数" in q)
    about_sorted = ("ソート" in q) or ("整列" in q) or ("単調" in q)
    return about_mid, about_complexity, about_sorted


def pick_followup_from_protocol(question: str, struct: dict, used_ids: list) -> tuple[str, str]:
    """
    論文寄せ:
    - 追い質問は「プロトコル（事前候補）」から選ぶ
    - struct（例/反例/境界の不足）は “どのプロトコルカテゴリを優先するか” に使う
    - used_ids があれば同一ループ内で重複回避
    return: (proto_id, raw_text)
    """
    about_mid, about_cx, about_sorted = _topic_flags(question)

    need_example = not bool(struct.get("has_example"))
    need_ce = not bool(struct.get("has_counterexample"))
    need_bd = not bool(struct.get("has_boundary"))

    # 優先カテゴリ（論文の inquiry protocol 的に「不足を埋める」）
    candidates = []

    if about_sorted and need_ce:
        candidates += FOLLOWUP_PROTOCOLS["counterexample"]
    if about_mid:
        candidates += FOLLOWUP_PROTOCOLS["mid_choice"]
    if about_cx:
        candidates += FOLLOWUP_PROTOCOLS["complexity"]

    # 足りない要素を優先
    if need_example:
        candidates += FOLLOWUP_PROTOCOLS["example_general"]
    if need_bd:
        candidates += FOLLOWUP_PROTOCOLS["boundary"]
    if need_ce:
        candidates += FOLLOWUP_PROTOCOLS["counterexample"]

    # 最後に “根拠” を初学者向けの言葉で聞く保険
    candidates += FOLLOWUP_PROTOCOLS["invariant_reason"]

    # 重複回避
    if FOLLOWUP_AVOID_REPEAT_IN_LOOP and used_ids:
        for pid, text in candidates:
            if pid not in used_ids:
                return pid, text

    # どれも使ってたら先頭でOK（シンプル）
    pid, text = candidates[0]
    return pid, text


# ---------- UI helper ----------
def _inject_css():
    st.markdown(
        """
<style>
/* 全体 */
.block-container { padding-top: 2.0rem; padding-bottom: 1.2rem; }
h1, h2, h3 { letter-spacing: -0.01em; }

/* カード */
/* ===== Dark theme friendly card ===== */
.card{
  background: rgba(128,128,128,0.18);
  color: var(--text-color) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 1px 12px rgba(0,0,0,0.25);
  margin-bottom: 12px;
}

.card h4{
  margin: 0 0 10px 0;
  font-size: 0.95rem;
  color: var(--text-color) !important;
}

/* 説明文 */
.small{
  color: rgba(255,255,255,0.72) !important;
  font-size: 0.88rem;
  line-height: 1.35;
}

/* 1行の間隔を詰めて“バランス”改善 */
.step{
  display: flex;
  gap: 12px;
  align-items: flex-start;
  margin: 10px 0;
}

/* バッジ */
.badge{
  width: 22px;
  height: 22px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  border: 1px solid rgba(255,255,255,0.25);
  color: rgba(255,255,255,0.75);
}

.badge.active{
  background: rgba(59,130,246,0.20);
  border-color: rgba(59,130,246,0.55);
  color: rgba(255,255,255,0.92);
  font-weight: 700;
}

.badge.done{
  background: rgba(16,185,129,0.18);
  border-color: rgba(16,185,129,0.45);
  color: rgba(255,255,255,0.88);
  font-weight: 700;
}

/* Mode pill */
.pill {
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(49,51,63,.15);
  background: rgba(49,51,63,.03);
  font-size: 0.85rem;
}
.pill b { font-weight: 700; }

/* Teaching helper box */
/* Teaching helper box (dark theme friendly) */
.helper {
  border-radius: 14px;
  padding: 12px 12px;
  margin-top: 10px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(128,128,128,0.10);
  color: var(--text-color) !important;
}

.helper.green {
  background: rgba(59,130,246,0.14);
  border-color: rgba(59,130,246,0.14);
}

.helper.red {
  background: rgba(239,68,68,0.14);
  border-color: rgba(239,68,68,0.35);
}

.helper .title {
  font-weight: 800;
  margin-bottom: 6px;
  color: var(--text-color) !important;
}

.helper .headline {
  font-weight: 900;
  font-size: 1.05rem;
  margin-bottom: 6px;
  color: var(--text-color) !important;
}

.helper.red .headline {
  text-decoration: underline;
}

/* ✅ ここが重要：helper内の small は “白固定” を上書きする */
.helper .small {
  color: rgba(255,255,255,0.82) !important;
}
/* 改善方針の枠 */
.panel {
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 14px;
  padding: 12px 12px 10px 12px;
  background: rgba(255,255,255,0.06);
  margin-top: 10px;
}
.panel .panel-title{
  font-size: 1.05rem;
  font-weight: 900;
  margin-bottom: 6px;
}
/* Teaching Helper / ブロックUIの文章を少し大きく */
.helper-note{
  font-size: 0.95rem;
  font-weight: 650;
  line-height: 1.45;
}
.helper-hint{
  font-size: 0.95rem;
  font-weight: 650;
  line-height: 1.45;
  margin-top: 6px;
}
.rewrite-wrap{
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 14px;
  padding: 12px 12px;
  background: rgba(255,255,255,0.06);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_objectives(current_obj: int):
    # Fig.2(A) 相当：目標ステップ（Understand→Apply→Analyze）:contentReference[oaicite:6]{index=6}
    steps = [
        ("理解の確認", "AlgoBoが概念を理解しているか確認する"),
        ("実装の支援", "AlgoBoが問題を解くコードを書けるようにする"),
        ("深掘り議論", "応用・類似手法・実世界例まで話す"),
    ]
    html = ['<div class="card"><h4>Learning Objectives</h4>']
    for i, (t, d) in enumerate(steps, start=1):
        cls = "active" if i == current_obj else ("done" if i < current_obj else "")
        html.append(
            "<div class='step'>"
            f"<div class='badge {cls}'>{i}</div>"
            "<div>"
            f"<div><b>{t}</b></div>"
            f"<div class='small'>{d}</div>"
            "</div></div>"
        )

    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)



def render_profile():
    # Fig.2(B) 相当：AlgoBoプロフィール:contentReference[oaicite:7]{index=7}
    st.markdown(
        """
<div class="card">
  <h4>AlgoBo Profile</h4>
  <div class="small">
    ・高校2年生（想定） / C言語の基礎は少し分かる<br/>
    ・二分探索で「なぜソートが必要？」「whileの範囲更新」が苦手<br/>
    ・答えをすぐ言わず、助けを求めたり質問したりする
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_problem():
    # Fig.2(C) 相当：問題表示（簡易）:contentReference[oaicite:8]{index=8}
    st.markdown(
        """
<div class="card">
  <h4>Problem</h4>
  <div class="small">
    <b>Binary Search (intro)</b><br/>
    昇順ソート済み配列 arr と target が与えられたとき、target の index を返す（なければ -1）。<br/>
    例: int arr[] = {1,3,5,7,9};, target=7 → 3
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

def render_left_panel_content():
    st.markdown(
        """
<div class="card">
  <h4>教える内容（Binary Search）</h4>
  <div class="small">
    ・二分探索とはなにか<br/>
    ・配列がソート済みでないといけない理由<br/>
    ・計算量について<br/>
    ・C言語による実装方法<br/>
  </div>
</div>

<div class="card">
  <h4>指導のポイント</h4>
  <div class="small">
    ・説明だけで終わらず、TeachbleAgentに「なぜ？」を言わせる<br/>
    ・<b>コードも教える</b>（関数、while条件、low/high更新、返り値の設計）<br/>
    ・必要なら「コードを書いてどう動く？」と質問して手を動かさせる
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_mode_pill(mode: str):
    label = "Help-receiver" if mode == "HELP_RECEIVER" else "Questioner"
    st.markdown(f'<span class="pill">Mode: <b>{label}</b></span>', unsafe_allow_html=True)


def render_teaching_helper_box(helper_state: dict):
    if not helper_state:
        return
    kind = helper_state.get("kind", "green")
    title = helper_state.get("title", "Teaching Helper")
    body = _ensure_str(helper_state.get("body", "") or "", label="helper.render.body")

    headline = first_sentence_ja(body)
    rest = body[len(headline):].strip() if body.startswith(headline) else body

    debug = helper_state.get("debug", "")

    rest_html = f'<div class="small">{rest}</div>' if rest else ""
    debug_html = ""  # 見せない運用

    # ✅ kindでアイコンを切り替え
    icon = "⚠️" if kind == "red" else "✅"   # ここを好きに

    st.markdown(
        f"""
<div class="helper {kind}">
  <div class="title">{title}</div>
  <div class="headline">{icon} {headline}</div>
  {rest_html}
  {debug_html}
</div>
        """,
        unsafe_allow_html=True,
    )



def main():
    st.set_page_config(page_title="TeachYou (Streamlit)", layout="wide")
    _inject_css()

    # -------- state init --------
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "二分探索って何ですか？どうしてソートが必要なんですか！"}
        ]
    if "knowledge_state" not in st.session_state:
        st.session_state.knowledge_state = {"facts": [], "code_implementation": []}
    if "mode" not in st.session_state:
        st.session_state.mode = "HELP_RECEIVER"
    if "msg_count" not in st.session_state:
        st.session_state.msg_count = 0
    if "tutor_history" not in st.session_state:
        st.session_state.tutor_history = []
    if "past_questions" not in st.session_state:
        st.session_state.past_questions = []
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "objective" not in st.session_state:
        st.session_state.objective = 1  # 1..3
    if "helper" not in st.session_state:
        st.session_state.helper = None
    if "helper_block" not in st.session_state:
        st.session_state.helper_block = False
    if "session_ended" not in st.session_state:
        st.session_state.session_ended = False
    if "final_log_json" not in st.session_state:
        st.session_state.final_log_json = None
    if "final_log_filename" not in st.session_state:
        st.session_state.final_log_filename = None

    if "tutor_text_history" not in st.session_state:
        st.session_state.tutor_text_history = []

    # Teaching Helper 表示管理（論文は6メッセージごと） :contentReference[oaicite:3]{index=3}
    if "helper_last_shown_msg_count" not in st.session_state:
        st.session_state.helper_last_shown_msg_count = 0

    if "helper_ephemeral_show" not in st.session_state:
        st.session_state.helper_ephemeral_show = False


    # constructive loop 管理（論文の「満足するまで追い質問」） :contentReference[oaicite:4]{index=4}
    if "question_loop_active" not in st.session_state:
        st.session_state.question_loop_active = False
    if "question_loop_round" not in st.session_state:
        st.session_state.question_loop_round = 0
    if "last_thinking_question" not in st.session_state:
        st.session_state.last_thinking_question = None

    if "loop_used_proto_ids" not in st.session_state:
        st.session_state.loop_used_proto_ids = []   # 追い質問ループ内で使ったプロトコルID


    # --- normalized log (turns) ---
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if "started_at" not in st.session_state:
        st.session_state.started_at = datetime.datetime.now().isoformat()
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "turn_id" not in st.session_state:
        st.session_state.turn_id = 0

    # pending: helper snapshot / choice (turnに紐づけるため)
    if "pending_helper_snapshot" not in st.session_state:
        st.session_state.pending_helper_snapshot = None
    if "pending_helper_choice" not in st.session_state:
        st.session_state.pending_helper_choice = None

    # ✅ ブロック中に保持する pending
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "pending_classification" not in st.session_state:
        st.session_state.pending_classification = None

    # 右ペイン等を使う場合に備えて（CHAT_ONLY=Trueでも安全）
    if "user_code" not in st.session_state:
        st.session_state.user_code = ""
    if "playground" not in st.session_state:
        st.session_state.playground = ""

    if "helper_chat_keys" not in st.session_state:
        st.session_state.helper_chat_keys = set()

    if "helper_log_keys" not in st.session_state:
      st.session_state.helper_log_keys = set()

    if "ui_busy" not in st.session_state:
        st.session_state.ui_busy = False
    if "ui_task" not in st.session_state:
        st.session_state.ui_task = None

    client = get_openai_client()
    if not client:
        st.warning("OpenAI APIキーが見つかりません。OPENAI_API_KEY を設定してください。")
        return

    # ---------- layout ----------
    if SHOW_LEFT_PANEL and SHOW_RIGHT_PANEL:
        left, mid, right = st.columns([1.15, 2.2, 1.65], gap="large")
    elif SHOW_LEFT_PANEL and not SHOW_RIGHT_PANEL:
        left, mid = st.columns([1.15, 2.2], gap="large")
        right = None
    else:
        st.markdown("<style>.block-container{max-width: 980px;}</style>", unsafe_allow_html=True)
        mid = st.container()
        left = None
        right = None

    if left is not None:
        with left:
            st.markdown("### ガイド")
            render_left_panel_content()
            # st.markdown("### Objectives")
            # render_objectives(st.session_state.objective)
            #render_profile()
            #render_problem()
            #st.markdown("<div style='padding:10px;border:2px solid red;'>TEST</div>", unsafe_allow_html=True)

            # with st.expander("Developer: Knowledge State / Logs", expanded=False):
            #     st.json(st.session_state.knowledge_state)

    if right is not None:
        with right:
            st.markdown("### Code")
            tab1, tab2, tab3 = st.tabs(["あなたの提出コード", "プレイグラウンド", "AlgoBoのコード"])
            with tab1:
                st.session_state.user_code = st.text_area(" ", st.session_state.user_code, height=220, label_visibility="collapsed")
            with tab2:
                st.session_state.playground = st.text_area(" ", st.session_state.playground, height=220, label_visibility="collapsed")
                st.caption("※論文UIの Code playground っぽい枠（実行機能は必要に応じて追加）")
            with tab3:
                codes = st.session_state.knowledge_state.get("code_implementation", [])
                if codes:
                    st.code("\n\n".join(codes[-3:]), language="C")
                else:
                    st.caption("AlgoBoのコードはまだありません。")

    # ---------- helpers ----------
    def _compute_antipattern(next_cls: Optional[str] = None, next_text: Optional[str] = None):
        W = HELPER_EVERY_N_MESSAGES

        cls_hist = st.session_state.tutor_history + ([next_cls] if next_cls else [])
        txt_hist = st.session_state.tutor_text_history + ([next_text] if next_text else [])

        if not cls_hist:
            return "Default", {"n": 0}

        # 直近W件（最大W件）
        window_cls = cls_hist[-W:]
        window_txt = txt_hist[-W:] if txt_hist else []
        n = len(window_cls)

        # ✅ 追加：窓が埋まるまで（n<W）は確定判定しない
        if n < W:
            return "Default", {
                "window": W,
                "n": n,
                "note": "warming_up",   # ログ解析用
                "dist": Counter(window_cls).most_common(6),
                "window_cls": window_cls,
            }

        RATIO_TH = 0.67
        # ✅ 変更：n ではなく W 基準で MIN_COUNT を作る
        MIN_COUNT = math.ceil(RATIO_TH * W)
        AVG_LEN_TH = 40

        def is_cmd_like(c: str) -> bool:
            return ("Commanding" in c) or ("Fixing" in c)

        def is_explaining(c: str) -> bool:
            return ("Comprehension" in c) or ("Elaboration" in c)

        low_content_cls = {"Encouraging", "Accepting/Reject", "Feedback"}
        def is_low_content(c: str) -> bool:
            return c in low_content_cls

        cmd_like_cnt = sum(1 for c in window_cls if is_cmd_like(c))
        explain_cnt  = sum(1 for c in window_cls if is_explaining(c))
        low_cnt      = sum(1 for c in window_cls if is_low_content(c))

        avg_len = (sum(len(t or "") for t in window_txt) / len(window_txt)) if window_txt else 0.0
        dist = Counter(window_cls).most_common(6)

        # 判定
        if cmd_like_cnt >= MIN_COUNT:
            anti = "Commanding"
        elif explain_cnt >= MIN_COUNT:
            anti = "Spoon-feeding"
        elif low_cnt >= MIN_COUNT and avg_len < AVG_LEN_TH:
            anti = "Under-teaching"
        else:
            anti = "Default"

        stats = {
            "window": W,
            "n": n,
            "ratio_th": RATIO_TH,
            "min_count": MIN_COUNT,
            "cmd_like_cnt": cmd_like_cnt,
            "explain_cnt": explain_cnt,
            "low_cnt": low_cnt,
            "avg_len": avg_len,
            "dist": dist,
            "window_cls": window_cls,
        }
        return anti, stats


    def _fmt_dist(stats: Optional[dict]) -> str:
        if not stats or not stats.get("n"):
            return ""
        n = stats["n"]
        dist = stats.get("dist", [])
        dist_str = ", ".join([f"{k}:{v}" for k, v in dist])
        return f"直近{n}件の分類分布: {dist_str}（cmd_like={stats.get('cmd_like_cnt')}, explain={stats.get('explain_cnt')}, low={stats.get('low_cnt')}）"


    def _set_helper_state(antipattern: str, next_turn_num: int, original_prompt: Optional[str] = None, original_classification: Optional[str] = None, ap_stats: Optional[dict] = None):
        # 論文寄せ：6メッセージごとにしか出さない :contentReference[oaicite:11]{index=11}
        if next_turn_num % HELPER_EVERY_N_MESSAGES != 0:
            st.session_state.helper = None
            st.session_state.helper_block = False
            st.session_state.helper_ephemeral_show = False  # ✅ 追加
            return

        # 表示したのでカウンタ更新（ログ用）
        st.session_state.helper_last_shown_msg_count = next_turn_num

        if antipattern == "Commanding":
            st.session_state.helper = {
                "kind": "red",
                "title": "Teaching Helper",
                "body": "指示（命令/修正）に偏っています。AlgoBoに『なぜ？』『根拠は？』を説明させる質問を混ぜましょう。",
                "options": [
                    "理由（why）を質問する",
                    "例や反例を求める",
                    "境界条件（low/high, mid更新）を一緒に確認する",
                ],
            }
            st.session_state.helper_block = True

        elif antipattern == "Spoon-feeding":
            st.session_state.helper = {
                "kind": "red",
                "title": "Teaching Helper",
                "body": "説明（答えの提示）に偏っています。理解確認や思考を促す質問に切り替えましょう。",
                "options": [
                    "理解チェック質問（今の理解を言ってもらう）",
                    "一部を穴埋めにして考えさせる",
                    "『もし〜なら？』の思考実験を投げる",
                ],
            }
            st.session_state.helper_block = True

        elif antipattern == "Under-teaching":
            st.session_state.helper = {
                "kind": "green",
                "title": "Teaching Helper",
                "body": "今のやり取りは情報が薄めかもです。短いヒントだけでなく、理由＋具体例（または反例）まで出すとAlgoBoが学びやすいです。",
                "options": [],
            }
            st.session_state.helper_block = False

        else:  # Default
            st.session_state.helper = {
                "kind": "green",
                "title": "Teaching Helper",
                "body": "良い流れです。『なぜソートが必要か』『low/high更新の根拠』を言語化させると深まります。",
                "options": [],
            }
            st.session_state.helper_block = False

        # # 直近Wの分布を helper に添付（ログ/表示用）
        # if st.session_state.helper and ap_stats:
        #     st.session_state.helper["debug"] = _fmt_dist(ap_stats)

        # ✅ 会話(messages)にも残す（rerunで重複しない）
        append_helper_to_chat(
            st.session_state.helper,
            turn_num=next_turn_num,
            antipattern=antipattern,
            blocked=bool(st.session_state.helper_block),
        )

        # ✅ 追加：表示した瞬間に logs にも残す
        log_helper_shown(
            turn_num=next_turn_num,
            helper_state=st.session_state.helper,
            blocked=bool(st.session_state.helper_block),
            antipattern=antipattern,
            ap_stats=ap_stats,
            original_prompt=(original_prompt if st.session_state.helper_block else None),
            original_classification=(original_classification if st.session_state.helper_block else None),
        )

        # ✅ Default/Under-teaching（ブロックなし・緑）のときだけ下ボックスを1回出す
        if antipattern in {"Default", "Under-teaching"} and (not st.session_state.helper_block):
            st.session_state.helper_ephemeral_show = True
        else:
            st.session_state.helper_ephemeral_show = False




    def build_rewrite_template(antipattern: Optional[str], choice: str, original: str, context: str = "") -> str:
        """
        固定テンプレだけだと文脈ズレが出るので、軽量ヒューリスティックで分岐する。
        - context: 直近会話（helper除外）
        - original: 未送信の元案
        """
        ctx = (context or "") + "\n" + (original or "")
        ctx_low = ctx.lower()
        about_log = ("log" in ctx_low) or ("対数" in ctx) or ("比較回数" in ctx) or ("計算量" in ctx) or ("1/3" in ctx)
        about_bounds = ("low" in ctx_low) or ("high" in ctx_low) or ("mid" in ctx_low) or ("while" in ctx_low) or ("境界" in ctx)

        # Spoon-feeding 用（説明しすぎ）
        if antipattern == "Spoon-feeding":
            if "理解チェック" in choice:
                if about_log:
                    return "いまの説明だと「log N」って言ってるけど、N が2倍になったら回数はどう変わるイメージですか？ 具体例で言えますか？"
                return "いまの二分探索テンプレで、low/high/mid がそれぞれ何を意味するか説明できますか？ その説明を聞いてから一緒にコードにしてみたいです。"
            if "穴埋め" in choice:
                if about_log:
                    return "回数が logN になる直感を穴埋めで言ってみましょう。N個から始めて、毎回『少なくとも ___ 倍』減るなら、何回で1個以下になりますか？"
                return "まずは穴埋めで考えてみましょう。\nlow, high = 0, n-1\nwhile low <= high:\n    mid = (low+high)//2\n    if a[mid] == x:\n        return mid\n    elif x < a[mid]:\n        high = ____\n    else:\n        low = ____\nこの2つの空欄は何になりますか？理由も教えてください。"
            if "思考実験" in choice:
                if about_log:
                    return "もし毎回「半分」じゃなくて「1/3」だけ減らせるとしたら、回数は logN のままですか？ 直感で説明できますか？"
                return "もし while 条件を low < high に変えたら、どんなケースで困りそうですか？ 逆に low <= high のままだと何が安心ですか？"

        # Commanding 用（命令・修正ばかり）
        if antipattern == "Commanding":
            if "理由" in choice:
                return "今の方針だと、なぜ探索範囲を半分にできると思いますか？「左/右を捨てていい根拠」を自分の言葉で言えますか？"
            if "例や反例" in choice:
                return "反例を考えてみたいです。ソートされていない配列で二分探索をしたら、どんな入力で間違えますか？"
            if "境界条件" in choice:
                return "境界条件を確認しよう！low と high を更新するとき、無限ループにならない条件って何だと思いますか？"

        # fallback
        return "今の理解を自分の言葉で説明してみてください。"



    def _process_turn(prompt_text: str, classification: str):
        answer_structure = None
        loop_end_reason = None

        # ===== timing / question generation trace =====
        msg_count_before = st.session_state.msg_count  # increment前
        timing = {
            "policy": None,              # "fixed_every_3" / "constructive_loop" など
            "rule": None,                # 例: "msg_count_after%3==0"
            "msg_count_before": msg_count_before,
            "msg_count_after": None,     # increment後に埋める
            "triggered": False,
            "trigger_kind": "none",      # "cycle_start" / "followup" / "none"
        }
        question_gen = None
        followup_gen = None


        # ★追加：このターンのAlgoBo発話に対応するメタ
        llm_meta = None
        llm_meta_source = None
        consistency_check = None

        # 追加：answered と next を分離して記録する
        answered_round = None
        answered_question = None
        next_round = None
        next_question = None

        # ✅ 実際に送信されるTutor発話だけ履歴に積む
        st.session_state.tutor_history.append(classification)
        st.session_state.tutor_text_history.append(prompt_text)

        prompt_text = _ensure_str(prompt_text, label="tutor.prompt")

        # Tutor message
        st.session_state.messages.append({"role": "user", "content": prompt_text})

        last_convo = build_llm_context(last_k=3)

        # このターンの helper 情報（ブロック発生時も保持する）
        # このターンの helper 情報（ブロック発生時も保持する）
        helper_snap = st.session_state.pending_helper_snapshot or {}

        original_text = helper_snap.get("original_text")
        original_cls  = helper_snap.get("original_classification")

        was_rewritten = bool(original_text) and (prompt_text != original_text)

        helper_info = {
            "antipattern": helper_snap.get("antipattern"),
            "blocked": bool(helper_snap.get("blocked", False)),
            "kind": helper_snap.get("kind"),
            "title": helper_snap.get("title"),
            "choice": st.session_state.pending_helper_choice,
            "body": _ensure_str(helper_snap.get("body", ""), label="helper_info.body"),
            "options": helper_snap.get("options", []),

            # ✅ 追加：介入の前後が turns に残る
            "original_prompt": original_text,                     # ブロック時の「元の送信案」
            "original_classification": original_cls,              # その分類
            "rewritten_prompt": prompt_text if was_rewritten else None,  # 実際に送った文（書き直しなら入る）
            "was_rewritten": was_rewritten,                       # 書き直し送信だったか
            "rewritten_classification": classification if was_rewritten else None,

            "final_prompt": prompt_text,
            "final_classification": classification,

            "antipattern_stats": helper_snap.get("antipattern_stats"),

        }

        # Reflect: 直前のTutor発話から知識を抽出し、既存の全知識との矛盾を確認する
        new_knowledge = op_extract(client, prompt_text) or []
        consistency_check = op_check_consistency(
            client,
            st.session_state.knowledge_state,
            new_knowledge,
        )
        if new_knowledge and consistency_check.get("verdict") != "CONFLICT":
            st.session_state.knowledge_state = op_update(
                client, st.session_state.knowledge_state, new_knowledge
            )

        # --- msg_count update ---
        st.session_state.msg_count += 1

        msg_count_after = st.session_state.msg_count
        timing["msg_count_after"] = msg_count_after

        # --- constructive loop（論文の「満足するまで追い質問」） :contentReference[oaicite:14]{index=14}
        # --- constructive loop（十分条件で終了、capは保険） ---
        if consistency_check and consistency_check.get("verdict") == "CONFLICT":
            st.session_state.mode = "HELP_RECEIVER"
            response_text = op_compose(client, new_knowledge, last_convo, consistency_check)
            llm_meta = st.session_state.get("_last_llm_meta")
            llm_meta_source = "compose(consistency_conflict)"

            st.session_state.question_loop_active = False
            st.session_state.question_loop_round = 0
            st.session_state.last_thinking_question = None
            st.session_state.loop_used_proto_ids = []
            loop_end_reason = "consistency_conflict"

            if LOG_TIMING_TRACE:
                timing["policy"] = "consistency_check"
                timing["rule"] = "consistency_check.verdict == CONFLICT"
                timing["triggered"] = True
                timing["trigger_kind"] = "consistency_conflict"

        elif st.session_state.question_loop_active and st.session_state.last_thinking_question:
            answered_round = st.session_state.question_loop_round
            answered_question = st.session_state.last_thinking_question

            struct = judge_answer_structure(client, answered_question, prompt_text, last_convo)
            answer_structure = struct  # turnsに入れる用

            # ★論文寄せ：Response Quality Classifier で「満足」を決める（structureは補助）
            quality = judge_answer_quality(client, answered_question, prompt_text, last_convo)  # GOOD/BAD
            struct["quality"] = quality  # ログに残す用（任意）

            st.session_state.logs.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "answer_structure",
                "question": answered_question,
                "answer": prompt_text,
                "structure": struct,
            })

            satisfied = (quality == "GOOD") if USE_QUALITY_FOR_LOOP_END else (struct.get("verdict") == "GOOD")

            if satisfied:
                response_text = summarize_answer_as_algobo(client, answered_question, prompt_text)
                llm_meta = st.session_state.get("_last_llm_meta")
                llm_meta_source = "summarize"

                st.session_state.mode = "HELP_RECEIVER"
                st.session_state.question_loop_active = False
                st.session_state.question_loop_round = 0
                st.session_state.last_thinking_question = None
                st.session_state.loop_used_proto_ids = []   # ★ループ終了でリセット
                loop_end_reason = "satisfied"

            else:
                next_r = st.session_state.question_loop_round + 1

                if next_r > QUESTION_LOOP_MAX_FOLLOWUPS:
                    response_text = summarize_answer_as_algobo(client, answered_question, prompt_text)
                    llm_meta = st.session_state.get("_last_llm_meta")
                    llm_meta_source = "summarize(cap_hit)"
                    response_text += " ただ、もう少し具体例か反例も見たいので、あとで教えてほしいです！"

                    st.session_state.mode = "HELP_RECEIVER"
                    st.session_state.question_loop_active = False
                    st.session_state.question_loop_round = 0
                    st.session_state.last_thinking_question = None
                    st.session_state.loop_used_proto_ids = []  # ★リセット
                    loop_end_reason = "cap_hit"

                else:
                    st.session_state.mode = "QUESTIONER"
                    st.session_state.question_loop_round = next_r

                    # ★論文寄せ：プロトコルから追い質問を選ぶ → paraphraseで文脈に合わせる
                    used = st.session_state.loop_used_proto_ids or []
                    proto_id, raw_follow = pick_followup_from_protocol(answered_question, struct, used)

                    if FOLLOWUP_USE_PARAPHRASE:
                        follow = paraphrase(client, raw_follow, last_convo)
                        llm_meta = st.session_state.get("_last_llm_meta")
                        llm_meta_source = "paraphrase_followup"
                    else:
                        follow = raw_follow
                        llm_meta = None
                        llm_meta_source = "followup_protocol"
                    response_text = follow

                    if LOG_TIMING_TRACE:
                        timing["policy"] = "constructive_loop"
                        timing["rule"] = "not_satisfied -> followup"
                        timing["triggered"] = True
                        timing["trigger_kind"] = "followup"

                    if LOG_QUESTION_GEN_TRACE:
                        followup_gen = {
                            "proto_id": proto_id,
                            "protocol_text": raw_follow,
                            "paraphrase_raw": st.session_state.get("_last_paraphrase_raw", "") if FOLLOWUP_USE_PARAPHRASE else "",
                            "final_question": follow,  # paraphrase後（またはそのまま）
                            "round": next_r,
                        }

                    # 使ったプロトコルIDを記録（同じループでの連発防止）
                    if FOLLOWUP_AVOID_REPEAT_IN_LOOP:
                        st.session_state.loop_used_proto_ids = used + [proto_id]

                    next_round = next_r
                    next_question = response_text

                    st.session_state.last_thinking_question = response_text
                    loop_end_reason = f"followup:{proto_id}"


        else:
            # --- 通常の Mode-shifting（3メッセージごと） :contentReference[oaicite:16]{index=16}
            if st.session_state.msg_count % 3 == 0:
                st.session_state.mode = "QUESTIONER"
                st.session_state.question_loop_active = True
                st.session_state.loop_used_proto_ids = []  # ★新しい質問ループ開始なのでリセット
                st.session_state.question_loop_round = 0

                qtype = "HOW" if st.session_state.objective >= 3 else "WHY"
                refined_q = generate_question_safe(client, last_convo, st.session_state.past_questions, qtype=qtype)
                if LOG_TIMING_TRACE:
                    timing["policy"] = "fixed_every_3"
                    timing["rule"] = "msg_count_after % 3 == 0"
                    timing["triggered"] = True
                    timing["trigger_kind"] = "cycle_start"

                if LOG_QUESTION_GEN_TRACE:
                    question_gen = st.session_state.get("_last_question_gen_trace")

                llm_meta = st.session_state.get("_last_llm_meta")
                llm_meta_source = "generate_question_safe"
                response_text = refined_q

                st.session_state.past_questions.append(refined_q)
                st.session_state.last_thinking_question = refined_q
                response_text = refined_q
                loop_end_reason = "start"   # ★追加

                # ★追加：開始した質問も turns に残す
                next_round = 0
                next_question = refined_q

                st.session_state.objective = min(3, st.session_state.objective + 1)

            else:
                if LOG_TIMING_TRACE:
                    timing["policy"] = "fixed_every_3"
                    timing["rule"] = "msg_count_after % 3 != 0"
                    timing["triggered"] = False
                    timing["trigger_kind"] = "none"

                st.session_state.mode = "HELP_RECEIVER"
                response_text = op_compose(client, new_knowledge, last_convo, consistency_check)
                llm_meta = st.session_state.get("_last_llm_meta")
                llm_meta_source = "compose"

        # ✅ 質問モードの出力を必ず1問に整形＋状態も同期
        if st.session_state.mode == "QUESTIONER":
            response_text = normalize_question(response_text)

            # 「次にTutorが答えるべき質問」を state に残しているので、ここで正規化結果に同期
            if st.session_state.question_loop_active:
                st.session_state.last_thinking_question = response_text

            # turns の next_question にも入れているので、ローカル変数があれば同期
            if next_question is not None:
                next_question = response_text

        response_text = _ensure_str(response_text, label="algobo.response")
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # ✅ 正規化した 1ターンレコードを追加
        st.session_state.turn_id += 1
        st.session_state.turns.append({
            "turn_id": st.session_state.turn_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "objective": st.session_state.objective,
            "mode": st.session_state.mode,
            "tutor": {
                "text": prompt_text,
                "classification": classification,
            },
            "helper": helper_info,
            "algobo": {
                "text": response_text,
            },
            "llm_meta": llm_meta,
            "llm_meta_source": llm_meta_source,
            "knowledge_delta": new_knowledge,
            "consistency_check": consistency_check,
            "answer_structure": answer_structure,
            "timing": (timing if LOG_TIMING_TRACE else None),
            "question_gen": (question_gen if LOG_QUESTION_GEN_TRACE else None),
            "followup_gen": (followup_gen if LOG_QUESTION_GEN_TRACE else None),
            "loop": {
                "answered_round": answered_round,
                "answered_question": answered_question,
                "next_round": next_round,
                "next_question": next_question,
                "end_reason": loop_end_reason,
            },
        })

        # 既存ログ（必要なら残す）
        st.session_state.logs.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "tutor_input": prompt_text,
            "classification": classification,
            "algobo_response": response_text,
            "llm_meta": llm_meta,
            "llm_meta_source": llm_meta_source,
            "mode": st.session_state.mode,
            "objective": st.session_state.objective,
            "knowledge_delta": new_knowledge,
            "consistency_check": consistency_check,
            "helper": helper_info,
        })

        # pending をクリア（次ターンに持ち越さない）
        st.session_state.pending_helper_snapshot = None
        st.session_state.pending_helper_choice = None



    def _finalize_session_and_prepare_download(reason: str = "exit"):
        if st.session_state.session_ended:
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"algobo_session_{ts}.json"

        # 終了イベントをログに追加
        st.session_state.logs.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "end_session",
            "reason": reason,
        })

        # 終了メッセージも会話に追加（messagesにも残す）
        st.session_state.messages.append({
            "role": "assistant",
            "content": "了解！ここで終了にします。ログを保存したのでダウンロードしてください！"
        })

        payload = {
            "meta": {
                "session_id": st.session_state.session_id,
                "started_at": st.session_state.started_at,
                "ended_at": datetime.datetime.now().isoformat(),
                "reason": reason,
                "model_id": OPENAI_MODEL_ID_GEN,
                "model_ids": MODEL_IDS,
            },
            "turns": st.session_state.turns,
            "final_knowledge_state": st.session_state.knowledge_state,
            # 任意：再現性のために残したいなら
            "messages": st.session_state.messages,
            "raw_logs": st.session_state.logs,
        }

        st.session_state.final_log_json = json.dumps(payload, ensure_ascii=False, indent=2)
        st.session_state.final_log_filename = filename
        st.session_state.session_ended = True

        # 入力待ちやブロック状態を解除
        st.session_state.helper_block = False
        st.session_state.helper = None
        st.session_state.pending_prompt = None
        st.session_state.pending_classification = None
        st.session_state.pending_helper_snapshot = None
        st.session_state.pending_helper_choice = None



    def _reset_session():
        for k in [
            "messages", "knowledge_state", "mode", "msg_count", "tutor_history",
            "past_questions", "logs", "objective", "helper", "helper_block",
            "helper_choice", "pending_prompt", "pending_classification",
            "session_ended", "final_log_json", "final_log_filename",
            "turns", "turn_id", "session_id", "started_at",
            "pending_helper_snapshot", "pending_helper_choice",
            "tutor_text_history",
            "helper_last_shown_msg_count",
            "question_loop_active",
            "question_loop_round",
            "last_thinking_question",
            "helper_chat_keys", "helper_log_keys",
            "loop_used_proto_ids",
            "rewrite_origin",
            "rewrite_editor",
            "helper_choice_radio",
            "tutor_input",
            "helper_ephemeral_show",  # ✅ 追加
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


    # ---------- Middle ----------
    with mid:
        st.markdown("### ➀Talk with Teachable Agent")
        render_mode_pill(st.session_state.mode)

        chat_slot = st.empty()
        helper_slot = st.empty()
        choice_slot = st.empty()

        def _render_chat_and_helper():
            # chat
            with chat_slot.container():

                # ✅ ブロック中なら「このターンの helper メッセージだけ」chat 側から隠す
                blocked_turn = None
                if st.session_state.helper_block:
                    blocked_turn = (st.session_state.pending_helper_snapshot or {}).get("turn_num")

                for msg in st.session_state.messages:
                    # 今ブロックしてる helper だけは chat 側に出さない（choice側で見せるため）
                    if msg.get("meta", {}).get("type") == "helper" and blocked_turn is not None:
                        if msg.get("meta", {}).get("turn") == blocked_turn:
                            continue

                    with st.chat_message(msg["role"]):
                        if msg.get("meta", {}).get("type") == "helper":
                            kind = msg.get("meta", {}).get("kind") or ("red" if str(msg.get("content","")).startswith("🟥") else "green")
                            helper_text = str(msg.get("content", "")).replace("\n", "  \n")  # ★強制改行

                            if kind == "red":
                                st.error(helper_text)
                            else:
                                st.info(helper_text)
                        else:
                            st.text(str(msg.get("content", "")))

            # helper box：Default/Under-teaching だけ「1回」表示
            if (
                st.session_state.helper
                and (not st.session_state.helper_block)
                and st.session_state.get("helper_ephemeral_show", False)
            ):
                with helper_slot.container():
                    render_teaching_helper_box(st.session_state.helper)

                # ✅ 次のrerunでは消える（1回表示）
                st.session_state.helper_ephemeral_show = False
            else:
                helper_slot.empty()

        # まず現状描画（入力欄より上）
        _render_chat_and_helper()

        # --- queued task executor（送信中表示→実処理） ---
        if st.session_state.get("ui_busy") and st.session_state.get("ui_task"):
            task = st.session_state.ui_task
            prog = st.progress(0, text="送信中…（画面を更新しています）")

            toast_msg = None
            toast_icon = None

            try:
                if task["type"] == "normal_send":
                    prompt_text = (task.get("prompt") or "").strip()

                    if prompt_text.lower() in EXIT_COMMANDS:
                        prog.progress(60, text="終了処理中…")
                        _finalize_session_and_prepare_download(reason=prompt_text)
                        toast_msg, toast_icon = "終了しました", "🛑"

                    else:
                        prog.progress(15, text="送信中…（分類中）")
                        classification = classify_message(client, prompt_text) or "Unknown"

                        prog.progress(35, text="送信中…（介入判定中）")
                        next_turn_num = st.session_state.msg_count + 1
                        antipattern, ap_stats = _compute_antipattern(classification, prompt_text)

                        _set_helper_state(
                            antipattern,
                            next_turn_num,
                            original_prompt=prompt_text,
                            original_classification=classification,
                            ap_stats=ap_stats,
                        )

                        st.session_state.pending_helper_snapshot = {
                            "turn_num": next_turn_num,
                            "antipattern": antipattern,
                            "antipattern_stats": ap_stats,
                            "blocked": bool(st.session_state.helper_block),
                            "kind": (st.session_state.helper or {}).get("kind"),
                            "title": (st.session_state.helper or {}).get("title"),
                            "body": _ensure_str((st.session_state.helper or {}).get("body", ""), label="pending_helper.body"),
                            "options": (st.session_state.helper or {}).get("options", []),
                            "original_text": prompt_text if st.session_state.helper_block else None,
                            "original_classification": classification if st.session_state.helper_block else None,
                        }
                        st.session_state.pending_helper_choice = None
                        st.session_state.pending_prompt = prompt_text
                        st.session_state.pending_classification = classification

                        if st.session_state.helper_block:
                            prog.progress(100, text="Teaching Helper が介入しました（未送信）")
                            toast_msg, toast_icon = "Teaching Helper が介入しました（未送信）", "🟥"
                            # ここで止める（書き直しUIへ）
                        else:
                            prog.progress(70, text="送信中…（AlgoBo返信生成中）")
                            p = st.session_state.pending_prompt
                            c = st.session_state.pending_classification
                            st.session_state.pending_prompt = None
                            st.session_state.pending_classification = None
                            _process_turn(p, c)

                            prog.progress(100, text="完了！")
                            toast_msg, toast_icon = "完了しました", "✅"

                elif task["type"] == "rewrite_send":
                    orig = (st.session_state.pending_helper_snapshot or {}).get("original_text") or ""
                    edited = (task["rewrite_text"].strip() != orig.strip())

                    st.session_state.pending_helper_choice = task.get("choice")
                    update_helper_chat_summary(
                        task.get("turn_num"),
                        task.get("choice"),
                        status=("書き直して送信" if edited else "そのまま送信")
                    )

                    new_cls = classify_message(client, task["rewrite_text"]) or "Unknown"
                    _process_turn(task["rewrite_text"], new_cls)

                    st.session_state.helper_block = False
                    st.session_state.helper = None

                    prog.progress(100, text="完了！")
                    toast_msg, toast_icon = "完了しました", "✅"

                elif task["type"] == "end_session":
                    prog.progress(60, text="終了処理中…")
                    _finalize_session_and_prepare_download(reason="end_button")
                    prog.progress(100, text="完了！")
                    toast_msg, toast_icon = "終了しました", "🛑"

                if toast_msg:
                    st.toast(toast_msg, icon=toast_icon)

            except Exception as e:
                st.session_state.ui_busy = False
                st.session_state.ui_task = None
                st.exception(e)
                st.stop()

            st.session_state.ui_busy = False
            st.session_state.ui_task = None
            st.rerun()



        # --- 終了後UI（ダウンロード & リセット）---
        if st.session_state.session_ended and st.session_state.final_log_json:
            st.success("セッションを終了しました。下からログをダウンロードできます。")

            st.download_button(
                label="📥 ログをダウンロード（JSON）",
                data=st.session_state.final_log_json,
                file_name=st.session_state.final_log_filename or "algobo_session.json",
                mime="application/json",
            )

            col_a, col_b = st.columns([1, 2])
            with col_a:
                if st.button("🔁 新しいセッションを開始", key="reset_session"):
                    _reset_session()

            # 終了したら入力欄は出さない
            st.stop()


        # ブロック中は入力不可
        prompt = st.chat_input(
            "Tutorとしてメッセージを入力…",
            key="tutor_input",
            disabled=(st.session_state.helper_block or st.session_state.session_ended or st.session_state.ui_busy),
        )

        # 送信処理
        if prompt:
            if st.session_state.get("ui_busy"):
                st.toast("いま送信中です…", icon="⏳")
            else:
                st.session_state.ui_busy = True
                st.session_state.ui_task = {"type": "normal_send", "prompt": prompt}
                st.rerun()

        # choice（ブロック時だけ）
        if st.session_state.helper_block and st.session_state.helper:
            options = st.session_state.helper.get("options", [])
            helper_snap = st.session_state.pending_helper_snapshot or {}
            original = helper_snap.get("original_text", "")

            with choice_slot.container():

                # ✅ 問題点（先頭1文を強調）
                helper_body = _ensure_str((st.session_state.helper or {}).get("body", "") or "", label="helper.choice.body")
                headline = first_sentence_ja(helper_body)
                rest = helper_body[len(headline):].strip() if helper_body.startswith(headline) else ""

                kind = (st.session_state.helper or {}).get("kind", "red")
                if kind == "red":
                    st.error(headline)
                else:
                    st.success(headline)

                if rest:
                    st.markdown(f'<div class="helper-note">{rest}</div>', unsafe_allow_html=True)

                st.markdown(
                    '<div class="helper-note">ここでの操作は未送信です。改善方針を選んで、下の文章を編集してから送信してください。</div>',
                    unsafe_allow_html=True
                )

                st.markdown('<div class="rewrite-wrap">', unsafe_allow_html=True)
                # ✅ text_area 初期化（ブロック画面に入った瞬間だけ）
                if st.session_state.get("rewrite_origin") != original:
                    st.session_state["rewrite_editor"] = original
                    st.session_state["rewrite_origin"] = original

                # ✅ form：これで「2回押し」になりにくい
                # ↓↓↓ ここを「radioはformの外」にする ↓↓↓

                # 改善方針（formの外に出すと、選択で即rerunされて説明が変わる）
                choice = None
                if options:
                    st.markdown(
                        '<div class="panel"><div class="panel-title">改善方針</div>',
                        unsafe_allow_html=True
                    )

                    choice = st.radio(
                        " ",
                        options,
                        index=0,
                        key="helper_choice_radio",
                        format_func=lambda x: HELPER_OPTION_LABELS.get(x, x),
                        label_visibility="collapsed",
                    )

                    # ✅ ここが「選択に応じて変わる説明」
                    hint = HELPER_OPTION_HINTS.get(choice, "")
                    if hint:
                        st.markdown(f'<div class="helper-hint">{hint}</div>', unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # formは「文章編集＋送信ボタン」だけにする
                with st.form("helper_rewrite_form", clear_on_submit=False):
                    # form内
                    rewrite_text = st.text_area(
                        "送信する文章（編集して送信）",
                        key="rewrite_editor",
                        height=160,
                        disabled=st.session_state.ui_busy,
                    )
                    submitted = st.form_submit_button("✍️ 編集して送信", disabled=st.session_state.ui_busy)
                    end_clicked = st.form_submit_button("🛑 終了してログを保存", disabled=st.session_state.ui_busy)

                # 中身（radioとかtext_areaとか）
                st.markdown('</div>', unsafe_allow_html=True)

                # formの外で処理（重要）
                if submitted:
                    if st.session_state.get("ui_busy"):
                        st.toast("いま送信中です…", icon="⏳")
                    else:
                        choice = st.session_state.get("helper_choice_radio", "")
                        turn_num = (st.session_state.pending_helper_snapshot or {}).get("turn_num")

                        st.session_state.ui_busy = True
                        st.session_state.ui_task = {
                            "type": "rewrite_send",
                            "rewrite_text": rewrite_text,
                            "choice": choice,
                            "turn_num": turn_num,
                        }
                        st.rerun()

                if end_clicked:
                    if st.session_state.get("ui_busy"):
                        st.toast("いま処理中です…", icon="⏳")
                    else:
                        st.session_state.ui_busy = True
                        st.session_state.ui_task = {"type": "end_session"}
                        st.rerun()

        else:
            choice_slot.empty()



if __name__ == "__main__":
    main()
