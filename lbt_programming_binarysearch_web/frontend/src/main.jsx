import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.error || data.detail || detail;
    } catch {
      // keep default detail
    }
    throw new Error(detail);
  }

  return response.json();
}

function MessageList({ messages }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <div className="messages" aria-live="polite">
      {messages.map((message, index) => (
        <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
          <div className="messageInner">
            <div className="avatar" aria-hidden="true">
              {message.role === "user" ? "U" : "A"}
            </div>
            <div className="messageContent">
              <div className="messageLabel">{message.role === "user" ? "User" : "AlgoBo"}</div>
              <div className="messageBody">{message.content}</div>
            </div>
          </div>
        </article>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

function Sidebar({ mode, apiReady }) {
  return (
    <aside className="sidebar">
      <h1>TeachYou</h1>

      <section>
        <h2>教える内容</h2>
        <ul>
          <li>二分探索とはなにか</li>
          <li>ソート済みである必要</li>
          <li>計算量</li>
          <li>C言語での実装</li>
        </ul>
      </section>

      <section>
        <h2>指導のポイント</h2>
        <ul>
          <li>説明だけで終わらせない</li>
          <li>AlgoBo に理解を確認する</li>
          <li>例や反例で考えさせる</li>
        </ul>
      </section>

      <dl className="statusList">
        <div>
          <dt>Mode</dt>
          <dd>{mode === "QUESTIONER" ? "Questioner" : "Help-receiver"}</dd>
        </div>
        <div>
          <dt>API</dt>
          <dd className={apiReady ? "ok" : "warn"}>{apiReady ? "接続可能" : "確認中"}</dd>
        </div>
      </dl>
    </aside>
  );
}

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [mode, setMode] = useState("HELP_RECEIVER");
  const [apiReady, setApiReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");

  const canSend = useMemo(() => Boolean(sessionId && input.trim() && !busy), [sessionId, input, busy]);

  async function startSession() {
    setBusy(true);
    setError("");
    try {
      const data = await apiFetch("/api/session", { method: "POST", body: "{}" });
      setSessionId(data.session_id);
      setMessages(data.messages);
      setMode(data.mode);
    } catch (err) {
      setError(`セッションを開始できませんでした: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    if (!canSend) return;

    const text = input.trim();
    setInput("");
    setBusy(true);
    setError("");

    try {
      const data = await apiFetch("/api/chat", {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      setMessages(data.messages);
      setMode(data.mode);
    } catch (err) {
      setError(`送信できませんでした: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    async function boot() {
      try {
        const health = await apiFetch("/api/health");
        setApiReady(Boolean(health.has_openai_api_key));
      } catch {
        setApiReady(false);
      }
      await startSession();
    }
    boot();
  }, []);

  return (
    <main className="appShell">
      <Sidebar mode={mode} apiReady={apiReady} />
      <section className="chatPanel">
        <header className="chatHeader">
          <div className="chatHeaderInner">
            <div>
            <h2>Talk with Teachable Agent</h2>
            <p>ユーザーは Tutor、AlgoBo は学習者です。</p>
            </div>
            <button type="button" onClick={startSession} disabled={busy}>
              新しいセッション
            </button>
          </div>
        </header>

        {error && <div className="errorBanner">{error}</div>}
        <MessageList messages={messages} />

        <div className="composerShell">
          <form className="composer" onSubmit={sendMessage}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              rows={2}
              placeholder="Tutorとしてメッセージを入力..."
              disabled={busy}
            />
            <button type="submit" disabled={!canSend} aria-label="送信">
              {busy ? "…" : "↑"}
            </button>
          </form>
          <div className="composerHint">Ctrl / Cmd + Enter で送信</div>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
