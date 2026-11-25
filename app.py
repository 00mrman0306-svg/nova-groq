
import os
import time
import secrets
from collections import defaultdict
from pathlib import Path

from flask import Flask, request, jsonify, session, render_template_string
from dotenv import load_dotenv
from importlib import import_module

try:
    _groq = import_module("groq")
except ModuleNotFoundError as exc:
    raise RuntimeError("The Groq SDK is required. Install it with `python -m pip install groq`.") from exc

Groq = _groq.Groq
RateLimitError = _groq.RateLimitError
AuthenticationError = _groq.AuthenticationError
APIConnectionError = _groq.APIConnectionError
APIStatusError = _groq.APIStatusError

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

if not FLASK_SECRET_KEY:
    raise RuntimeError("FLASK_SECRET_KEY is missing from .env")

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.getenv("RENDER")),
    PERMANENT_SESSION_LIFETIME=86400
)

client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-120b"

PERSONALITY = """
You are NOVA, an AI assistant created by NV STUDIOS.

PERSONALITY:
- Friendly, intelligent and helpful.
- Explain things clearly.
- Be accurate and honest.
- Never invent facts.
- Adapt your response length to the question.
- Use a natural conversational tone.

GET MODE:
- Be more casual and playful.
- Occasionally use words like bro.
- Keep simple answers short.
- Use emojis sparingly.

SAFETY:
- Prioritize human safety.
- Never facilitate harmful or illegal activity.
- Be respectful and age-appropriate.
"""

# In-memory chat storage.
# Chats reset when the server restarts.

chats = defaultdict(list)
request_history = defaultdict(list)

MAX_REQUESTS_PER_MINUTE = 10
MAX_REQUESTS_PER_HOUR = 60
MAX_MESSAGES = 20
MAX_MESSAGE_LENGTH = 6000


def get_owner_id():
    session.permanent = True

    if "nova_owner_id" not in session:
        session["nova_owner_id"] = secrets.token_urlsafe(32)

    return session["nova_owner_id"]


def is_rate_limited(owner_id):
    now = time.time()
    events = request_history[owner_id]

    events[:] = [
        timestamp for timestamp in events
        if now - timestamp < 3600
    ]

    minute_count = sum(
        1 for timestamp in events
        if now - timestamp < 60
    )

    if minute_count >= MAX_REQUESTS_PER_MINUTE:
        return True

    if len(events) >= MAX_REQUESTS_PER_HOUR:
        return True

    events.append(now)
    return False


def generate_response(owner_id, message):
    history = chats[owner_id]

    messages = [
        {
            "role": "system",
            "content": PERSONALITY
        }
    ]

    messages.extend(history[-MAX_MESSAGES:])

    messages.append({
        "role": "user",
        "content": message
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=2048
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The AI returned an empty response.")

    # Save conversation only after a successful response.
    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    # Keep only the most recent messages.
    chats[owner_id] = history[-MAX_MESSAGES:]

    return answer


# ==========================================
# WEBSITE INTERFACE
# ==========================================

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOVA AI — NV STUDIOS</title>

<style>
:root {
    --bg: #080d12;
    --panel: #0e171e;
    --panel2: #121f27;
    --border: rgba(130, 210, 200, .12);
    --accent: #54dec4;
    --accent2: #20a995;
    --text: #e8f2f1;
    --muted: #849b9a;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    background:
        radial-gradient(ellipse at 50% -20%, #163c3b 0%, transparent 45%),
        var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
    height: 100vh;
    overflow: hidden;
}

.app {
    display: flex;
    height: 100vh;
}

/* SIDEBAR */

.sidebar {
    width: 255px;
    background: rgba(8, 16, 21, .92);
    border-right: 1px solid var(--border);
    padding: 24px 17px;
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo {
    width: 43px;
    height: 43px;
    border-radius: 14px;
    background: linear-gradient(145deg, #69efda, #168c83);
    display: grid;
    place-items: center;
    color: #061713;
    font-size: 23px;
    font-weight: 900;
    box-shadow: 0 0 25px rgba(65, 225, 196, .18);
}

.brand h1 {
    font-size: 19px;
    letter-spacing: 2px;
}

.brand small {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 2px;
}

.new-chat {
    padding: 13px;
    border-radius: 12px;
    border: 1px solid rgba(84, 222, 196, .35);
    background: rgba(84, 222, 196, .08);
    color: var(--accent);
    cursor: pointer;
    font-size: 14px;
    transition: .2s;
}

.new-chat:hover {
    background: rgba(84, 222, 196, .17);
    transform: translateY(-1px);
}

.sidebar-label {
    color: #69817f;
    font-size: 10px;
    letter-spacing: 2px;
}

.sidebar-info {
    margin-top: auto;
    padding: 14px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 13px;
}

.sidebar-info strong {
    font-size: 12px;
    color: var(--accent);
}

.sidebar-info p {
    margin-top: 7px;
    font-size: 11px;
    color: var(--muted);
    line-height: 1.6;
}

/* MAIN */

.main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
}

.topbar {
    height: 70px;
    flex-shrink: 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    background: rgba(9, 17, 22, .65);
}

.topbar-title {
    font-size: 13px;
    letter-spacing: 1px;
}

.status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--muted);
}

.status-dot {
    width: 7px;
    height: 7px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 12px var(--accent);
}

.messages {
    flex: 1;
    overflow-y: auto;
    padding: 35px 18px;
    scroll-behavior: smooth;
}

.message-list {
    max-width: 850px;
    margin: auto;
    display: flex;
    flex-direction: column;
    gap: 23px;
}

.welcome {
    margin: auto;
    max-width: 600px;
    text-align: center;
    padding: 70px 20px;
    animation: appear .6s ease;
}

.welcome-icon {
    font-size: 46px;
    color: var(--accent);
    text-shadow: 0 0 35px rgba(84, 222, 196, .3);
    margin-bottom: 22px;
}

.welcome h2 {
    font-size: clamp(28px, 5vw, 43px);
    letter-spacing: -1.5px;
    margin-bottom: 12px;
}

.welcome h2 span {
    color: var(--accent);
}

.welcome p {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.message {
    display: flex;
    gap: 13px;
    animation: appear .25s ease;
}

.avatar {
    width: 34px;
    height: 34px;
    flex-shrink: 0;
    border-radius: 11px;
    display: grid;
    place-items: center;
    font-size: 12px;
    font-weight: bold;
}

.ai-avatar {
    background: rgba(84, 222, 196, .14);
    color: var(--accent);
}

.user-avatar {
    background: #25353e;
    color: #d4e6e3;
}

.message-content {
    min-width: 0;
    flex: 1;
    padding-top: 4px;
}

.message-name {
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
}

.message-text {
    color: #cbd9d7;
    font-size: 14px;
    line-height: 1.8;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.user-message .message-text {
    color: #f0f7f6;
}

.typing {
    color: var(--accent);
    font-size: 13px;
    animation: pulse 1s infinite;
}

/* INPUT */

.composer-area {
    padding: 16px 20px 22px;
    background: linear-gradient(transparent, var(--bg) 25%);
}

.composer {
    max-width: 850px;
    margin: auto;
    background: #101c23;
    border: 1px solid rgba(84, 222, 196, .22);
    border-radius: 17px;
    display: flex;
    align-items: flex-end;
    padding: 10px 12px 10px 17px;
    gap: 12px;
    transition: .2s;
}

.composer:focus-within {
    border-color: rgba(84, 222, 196, .65);
    box-shadow: 0 0 25px rgba(84, 222, 196, .06);
}

textarea {
    flex: 1;
    min-width: 0;
    max-height: 150px;
    resize: none;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text);
    font-family: inherit;
    font-size: 14px;
    line-height: 1.6;
    padding: 7px 0;
}

textarea::placeholder {
    color: #6d8583;
}

.send {
    width: 39px;
    height: 39px;
    flex-shrink: 0;
    border: none;
    border-radius: 12px;
    background: var(--accent);
    color: #061714;
    font-size: 18px;
    font-weight: bold;
    cursor: pointer;
    transition: .2s;
}

.send:hover {
    background: #8af5e1;
    transform: scale(1.05);
}

.send:disabled {
    opacity: .4;
    cursor: not-allowed;
    transform: none;
}

.footer-note {
    text-align: center;
    color: #607573;
    font-size: 10px;
    margin-top: 12px;
}

@keyframes appear {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    50% { opacity: .35; }
}

@media(max-width: 650px) {
    .sidebar {
        display: none;
    }

    .topbar {
        padding: 0 17px;
    }

    .messages {
        padding: 25px 14px;
    }

    .composer-area {
        padding: 12px 12px 17px;
    }
}
</style>
</head>

<body>
<div class="app">

    <aside class="sidebar">
        <div class="brand">
            <div class="logo">✦</div>
            <div>
                <h1>NOVA AI</h1>
                <small>BY NV STUDIOS</small>
            </div>
        </div>

        <button class="new-chat" onclick="newChat()">
            + &nbsp; New conversation
        </button>

        <div class="sidebar-label">YOUR WORKSPACE</div>

        <div class="sidebar-info">
            <strong>✦ NOVA ENGINE</strong>
            <p>Powered by Groq<br>GPT-OSS 120B</p>
        </div>

        <div class="sidebar-info">
            <strong>NV STUDIOS</strong>
            <p>Built with creativity.<br>Designed for ideas.</p>
        </div>
    </aside>

    <main class="main">

        <header class="topbar">
            <div class="topbar-title">NOVA / CHAT</div>

            <div class="status">
                <span class="status-dot"></span>
                AI ENGINE CONNECTED
            </div>
        </header>

        <section class="messages" id="messages">
            <div class="message-list" id="messageList"></div>
        </section>

        <div class="composer-area">
            <form class="composer" id="chatForm">
                <textarea
                    id="userInput"
                    rows="1"
                    maxlength="6000"
                    placeholder="Message NOVA..."
                    required
                ></textarea>

                <button class="send" id="sendButton" type="submit">
                    ↑
                </button>
            </form>

            <div class="footer-note">
                NOVA can make mistakes. Verify important information.
                &nbsp; • &nbsp; NV STUDIOS
            </div>
        </div>

    </main>
</div>

<script>
const messageList = document.getElementById("messageList");
const messagesArea = document.getElementById("messages");
const input = document.getElementById("userInput");
const form = document.getElementById("chatForm");
const sendButton = document.getElementById("sendButton");

let busy = false;

function showWelcome() {
    messageList.innerHTML = "";

    const welcome = document.createElement("div");
    welcome.className = "welcome";

    const icon = document.createElement("div");
    icon.className = "welcome-icon";
    icon.textContent = "✦";

    const heading = document.createElement("h2");
    heading.append("Welcome to ");
    const span = document.createElement("span");
    span.textContent = "NOVA.";
    heading.appendChild(span);

    const paragraph = document.createElement("p");
    paragraph.textContent =
        "Your AI workspace. Ask questions, explore ideas, " +
        "write, learn and create.";

    welcome.append(icon, heading, paragraph);
    messageList.appendChild(welcome);
}

function addMessage(role, content) {
    const welcome = messageList.querySelector(".welcome");
    if (welcome) welcome.remove();

    const wrapper = document.createElement("div");
    wrapper.className = "message " +
        (role === "user" ? "user-message" : "ai-message");

    const avatar = document.createElement("div");
    avatar.className = "avatar " +
        (role === "user" ? "user-avatar" : "ai-avatar");
    avatar.textContent = role === "user" ? "YOU" : "✦";

    const contentBox = document.createElement("div");
    contentBox.className = "message-content";

    const name = document.createElement("div");
    name.className = "message-name";
    name.textContent = role === "user" ? "You" : "NOVA";

    const text = document.createElement("div");
    text.className = "message-text";
    text.textContent = content;

    contentBox.append(name, text);
    wrapper.append(avatar, contentBox);
    messageList.appendChild(wrapper);

    messagesArea.scrollTop = messagesArea.scrollHeight;

    return wrapper;
}

function showTyping() {
    const wrapper = addMessage("assistant", "NOVA is thinking...");
    wrapper.querySelector(".message-text").className =
        "message-text typing";
    return wrapper;
}

async function sendMessage(message) {
    if (busy || !message.trim()) return;

    busy = true;
    sendButton.disabled = true;

    addMessage("user", message);
    input.value = "";

    const typing = showTyping();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        typing.remove();

        if (!response.ok) {
            addMessage("assistant", data.error || "Something went wrong.");
        } else {
            addMessage("assistant", data.response);
        }

    } catch (error) {
        typing.remove();
        addMessage(
            "assistant",
            "Unable to connect to NOVA. Please try again."
        );
    }

    busy = false;
    sendButton.disabled = false;
    input.focus();
}

form.addEventListener("submit", function(event) {
    event.preventDefault();
    sendMessage(input.value);
});

input.addEventListener("keydown", function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

input.addEventListener("input", function() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 150) + "px";
});

async function newChat() {
    if (busy) return;

    try {
        const response = await fetch("/new-chat", {
            method: "POST"
        });

        if (!response.ok) {
            alert("Could not create a new conversation.");
            return;
        }

        showWelcome();
        input.value = "";
        input.focus();

    } catch (error) {
        alert("Unable to connect to NOVA.");
    }
}

showWelcome();

const previousMessages = {{ messages|tojson }};

previousMessages.forEach(function(message) {
    addMessage(message.role, message.content);
});

</script>
</body>
</html>
"""


# ==========================================
# FLASK ROUTES
# ==========================================

@app.route("/")
def home():
    owner_id = get_owner_id()

    return render_template_string(
        PAGE,
        messages=chats[owner_id],
        model=MODEL
    )


@app.route("/chat", methods=["POST"])
def chat():
    owner_id = get_owner_id()

    if is_rate_limited(owner_id):
        return jsonify({
            "error": "You've reached NOVA's request limit. Please wait a little."
        }), 429

    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str):
        return jsonify({"error": "Invalid message."}), 400

    message = message.strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "error": "Your message is too long. Please shorten it."
        }), 400

    try:
        answer = generate_response(owner_id, message)

        return jsonify({
            "response": answer,
            "model": MODEL
        })

    except RateLimitError:
        return jsonify({
            "error": "NOVA has reached its Groq API usage limit. Please try again later."
        }), 429

    except AuthenticationError:
        app.logger.exception("Groq authentication failed")

        return jsonify({
            "error": "NOVA's AI connection needs attention. Please try again later."
        }), 503

    except APIConnectionError:
        app.logger.exception("Groq connection failed")

        return jsonify({
            "error": "NOVA could not connect to its AI provider."
        }), 503

    except APIStatusError:
        app.logger.exception("Groq API returned an error")

        return jsonify({
            "error": "NOVA's AI provider returned an error. Please try again later."
        }), 502

    except Exception:
        app.logger.exception("Unexpected NOVA error")

        return jsonify({
            "error": "NOVA encountered an unexpected error."
        }), 500


@app.route("/new-chat", methods=["POST"])
def new_chat():
    owner_id = get_owner_id()
    chats[owner_id] = []

    return jsonify({"success": True})


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
