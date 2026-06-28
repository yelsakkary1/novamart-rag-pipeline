"""
FIXED VERSION — key changes marked with # FIX comments
The core idea: split each handler into:
  1. An "ack" function — returns instantly, tells Slack "got it"
  2. A "lazy" function — does the slow RAG/Claude work in the background
"""

import os
import re
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from query import query_with_history

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    # FIX: process_before_response is required for lazy listeners to work correctly
    process_before_response=True,
)

conversation_history = {}
MAX_HISTORY = 10


def get_history(user_id: str) -> list:
    return conversation_history.get(user_id, [])


def update_history(user_id: str, role: str, content: str):
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": role, "content": content})
    if len(conversation_history[user_id]) > MAX_HISTORY:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY:]


def clear_history(user_id: str):
    conversation_history[user_id] = []


# ─────────────────────────────────────────────
# DIRECT MESSAGES — split into ack + lazy
# ─────────────────────────────────────────────
def ack_dm(ack):
    # FIX: this returns almost instantly, so Slack never times out waiting
    ack()


def lazy_dm(message, say, client):
    # FIX: this runs in a background thread, can take as long as it needs
    user_id = message.get("user")
    text = message.get("text", "").strip()
    channel = message.get("channel")

    if message.get("bot_id"):
        return

    if text.lower() in ["reset", "clear", "start over"]:
        clear_history(user_id)
        say("Conversation reset. Ask me anything about NovaMart operations.")
        return

    client.reactions_add(channel=channel, name="thinking_face", timestamp=message["ts"])

    history = get_history(user_id)
    update_history(user_id, "user", text)

    answer = query_with_history(text, history)  # slow call — now safely isolated

    update_history(user_id, "assistant", answer)
    client.reactions_remove(channel=channel, name="thinking_face", timestamp=message["ts"])
    say(answer)


# FIX: register both halves — ack runs first, lazy runs after, in a thread pool
app.message()(ack=ack_dm, lazy=[lazy_dm])


# ─────────────────────────────────────────────
# @MENTIONS — same split
# ─────────────────────────────────────────────
def ack_mention(ack):
    ack()


def lazy_mention(event, say, client):
    user_id = event.get("user")
    text = event.get("text", "")
    channel = event.get("channel")

    clean_text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

    if not clean_text:
        say("Hey! Ask me anything about NovaMart store operations.")
        return

    client.reactions_add(channel=channel, name="thinking_face", timestamp=event["ts"])

    history = get_history(user_id)
    update_history(user_id, "user", clean_text)
    answer = query_with_history(clean_text, history)
    update_history(user_id, "assistant", answer)

    client.reactions_remove(channel=channel, name="thinking_face", timestamp=event["ts"])
    say(f"<@{user_id}> {answer}")


app.event("app_mention")(ack=ack_mention, lazy=[lazy_mention])


# ─────────────────────────────────────────────
# WELCOME — unchanged, this one is fast already
# ─────────────────────────────────────────────
@app.event("member_joined_channel")
def welcome(event, say):
    say(
        f"<@{event['user']}> Welcome! I'm the NovaMart AI Operations Assistant. "
        f"Ask me anything about store performance, inventory, labor costs, or customer satisfaction. "
        f"Type *reset* at any time to start a new conversation."
    )


if __name__ == "__main__":
    print("\nNovaMart AI Operations Assistant")
    print("-" * 40)
    print("Slack bot starting (with lazy listeners)...")

    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()