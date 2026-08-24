"""
app.py
Professional Streamlit chat UI — user right (blue), bot left (white).

Run with:
    streamlit run app.py
"""

import logging
import uuid
import json
import streamlit as st

from agent.agent import chat, AgentResponse
from agent.config import DEBUG

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

st.set_page_config(
    page_title="Aster & Row — Support",
    page_icon="🎒",
    layout="centered",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
  }

  .stApp { background-color: #f0f2f5; }
  #MainMenu, footer, header { visibility: hidden; }

  .block-container {
    max-width: 800px;
    padding-top: 1.5rem;
    padding-bottom: 6rem;
  }

  /* ── Top bar ── */
  .topbar {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .topbar-title  { font-size: 1.05rem; font-weight: 600; color: #111; margin: 0; }
  .topbar-sub    { font-size: 0.78rem; color: #888; margin: 0; }
  .online-dot    {
    width: 9px; height: 9px; background: #22c55e;
    border-radius: 50%; display: inline-block; margin-right: 5px;
  }

  /* ── Chat container ── */
  .chat-area {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 4px 0;
  }

  /* ── Shared bubble base ── */
  .bubble-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
  }

  /* USER — right side, blue */
  .bubble-row.user {
    flex-direction: row-reverse;
  }
  .bubble.user {
    background: #2563eb;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 15px;
    max-width: 72%;
    font-size: 0.93rem;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(37,99,235,0.25);
  }

  /* BOT — left side, white */
  .bubble-row.bot {
    flex-direction: row;
  }
  .bubble.bot {
    background: #ffffff;
    color: #1a1a1a;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 15px;
    max-width: 78%;
    font-size: 0.93rem;
    line-height: 1.7;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .bubble.bot p  { margin: 0 0 0.4rem 0; }
  .bubble.bot p:last-child { margin-bottom: 0; }
  .bubble.bot strong { color: #111; }

  /* Avatars */
  .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
  }
  .avatar.user { background: #1d4ed8; }
  .avatar.bot  { background: #f3f4f6; border: 1px solid #e0e0e0; }

  /* Handoff banner */
  .handoff-banner {
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    padding: 6px 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #92400e;
    margin-top: 6px;
  }

  /* ── Chat input ── */
  .stChatInputContainer {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
  }
  .stChatInputContainer textarea {
    font-size: 0.93rem !important;
    color: #111 !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stChatInputContainer textarea::placeholder { color: #9ca3af !important; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
  }
  [data-testid="stSidebar"] * { color: #333 !important; }
  .sidebar-label {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #9ca3af !important; margin: 1.1rem 0 0.4rem 0;
  }
  .sidebar-topic { font-size: 0.84rem; color: #555 !important; padding: 3px 0; }

  .stButton > button {
    background: #f9fafb !important; border: 1px solid #e5e7eb !important;
    color: #374151 !important; border-radius: 8px !important;
    font-size: 0.83rem !important; font-weight: 500 !important;
  }
  .stButton > button:hover {
    background: #f3f4f6 !important; border-color: #d1d5db !important;
  }

  .stToggle label { font-size: 0.83rem !important; color: #555 !important; }
  .stCaption { color: #bbb !important; font-size: 0.71rem !important; }
  hr { border-color: #f0f0f0 !important; margin: 0.7rem 0 !important; }

  /* Debug */
  .debug-box {
    background: #1e1e1e; color: #d4d4d4; padding: 0.8rem;
    border-radius: 8px; font-family: monospace; font-size: 0.73rem;
    max-height: 240px; overflow-y: auto; margin-top: 6px;
  }
  .streamlit-expanderHeader { font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎒 Aster & Row")
    st.markdown(
        '<span class="online-dot"></span>'
        '<span style="font-size:0.82rem;color:#555;">Support Agent · Online</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sidebar-label">I can help with</div>', unsafe_allow_html=True)
    for topic in [
        "📦  Returns & Refunds",
        "🚚  Shipping & Tracking",
        "🔧  Warranty Claims",
        "📋  Order Changes",
        "💳  Gift Cards",
        "🏕️  TrailPlus Membership",
    ]:
        st.markdown(f'<div class="sidebar-topic">{topic}</div>', unsafe_allow_html=True)

    st.divider()
    debug_mode = st.toggle("Debug mode", value=DEBUG)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Session `{st.session_state.session_id[:8]}`")

# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <span style="font-size:1.6rem;">🎒</span>
  <div>
    <p class="topbar-title">Aster &amp; Row Support</p>
    <p class="topbar-sub">Ask about orders, returns, shipping, warranties, and more</p>
  </div>
</div>
""", unsafe_allow_html=True)


def render_message(role: str, content: str, handoff: bool = False, debug_trace: dict = None):
    """Render a single chat bubble with proper left/right alignment."""
    import html as html_lib
    safe_content = content.replace("\n", "<br>")

    if role == "user":
        st.markdown(f"""
        <div class="bubble-row user">
          <div class="avatar user">👤</div>
          <div class="bubble user">{safe_content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        handoff_html = ""
        if handoff:
            handoff_html = (
                '<div class="handoff-banner">⚠️ <strong>Human support recommended</strong> — '
                'please contact our support team for further help.</div>'
            )
        st.markdown(f"""
        <div class="bubble-row bot">
          <div class="avatar bot">🎒</div>
          <div class="bubble bot">{safe_content}{handoff_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if debug_mode and debug_trace:
            with st.expander("Debug trace"):
                st.markdown(
                    f'<div class="debug-box"><pre>{json.dumps(debug_trace, indent=2, default=str)}</pre></div>',
                    unsafe_allow_html=True,
                )


# ── Render history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(
        role=msg["role"],
        content=msg["content"],
        handoff=msg.get("handoff", False),
        debug_trace=msg.get("debug_trace"),
    )

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your question here…"):

    # Show user bubble immediately
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner(""):
        try:
            response: AgentResponse = chat(
                user_message=prompt,
                session_id=st.session_state.session_id,
            )

            render_message(
                role="assistant",
                content=response.answer,
                handoff=response.handoff_recommended,
                debug_trace=response.debug_trace if debug_mode else None,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "handoff": response.handoff_recommended,
                "debug_trace": response.debug_trace if debug_mode else {},
            })

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                error_msg = (
                    "I'm temporarily unavailable due to high demand — the AI service has hit its "
                    "rate limit. Please wait a minute and try again. If this keeps happening, "
                    "our support team is always happy to help directly."
                )
            elif "401" in err or "403" in err or "API_KEY" in err or "permission" in err.lower():
                error_msg = (
                    "There's a configuration issue on our end. Please try again shortly, "
                    "or contact our support team if the problem persists."
                )
            elif "timeout" in err.lower() or "deadline" in err.lower():
                error_msg = (
                    "The request timed out — please try again. "
                    "If you keep seeing this, try rephrasing your question."
                )
            else:
                error_msg = (
                    "Something went wrong on my end. Please try again, "
                    "or reach out to our support team directly."
                )
            render_message("assistant", error_msg, handoff=False)
            logging.exception("Agent error")
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "handoff": False,
            })
