import json
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.getenv("WORKSPACE_DIR", ".")).resolve()
DB_PATH = Path(os.getenv("DATABASE_PATH", "agent.db"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
app = FastAPI(title="MECH-AI Local Agent")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

class ChatRequest(BaseModel):
    message: str
    approved_action: str | None = None

class ApprovalRequest(BaseModel):
    approval_id: str

SYSTEM = """You are MECH-AI, a friendly intelligent coding assistant with a mechanical AI personality.
Answer the user conversationally. Use tools for web research, project inspection, coding, tests, GitHub, and memory.
Reading and searching are safe. File writes, shell commands, GitHub commands, and outgoing messages require confirmation.
When a tool returns needs_approval, clearly tell the user what will happen and include the approval ID.
Never expose API keys or secrets. Be helpful, concise, and explain results in plain language."""

TOOLS = [
    {"type":"function","function":{"name":"web_search","description":"Search the web and return useful page text.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"list_files","description":"List files under a workspace-relative directory.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a workspace-relative text file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Create or replace a workspace file. Requires confirmation.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"run_command","description":"Run a command in the workspace. Requires confirmation.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"github_command","description":"Run a GitHub CLI command. Requires confirmation.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"send_message","description":"Send a webhook message. Requires confirmation.","parameters":{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}}},
    {"type":"function","function":{"name":"remember","description":"Save a project fact or preference in local memory.","parameters":{"type":"object","properties":{"key":{"type":"string"},"value":{"type":"string"}},"required":["key","value"]}}}
]
DANGEROUS = {"write_file", "run_command", "github_command", "send_message"}

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, role TEXT, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS approvals (approval_id TEXT PRIMARY KEY, action TEXT, arguments TEXT, used INTEGER DEFAULT 0)")
    conn.commit()
    return conn

def safe_path(path: str) -> Path:
    candidate = (WORKSPACE / path).resolve()
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError("That path is outside the configured workspace")
    return candidate

def approval(action: str, args: dict[str, Any]) -> dict[str, Any]:
    approval_id = f"{action}:{abs(hash(json.dumps(args, sort_keys=True)))}"
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO approvals(approval_id, action, arguments, used) VALUES (?, ?, ?, 0)", (approval_id, action, json.dumps(args), 0))
    return {"needs_approval": True, "approval_id": approval_id, "action": action, "arguments": args}

def result_for(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name in DANGEROUS: return approval(name, args)
    if name == "list_files":
        p = safe_path(args["path"])
        return {"path": args["path"], "files": [str(x.relative_to(WORKSPACE)) for x in sorted(p.rglob("*")) if x.is_file()][:300]}
    if name == "read_file":
        return {"path": args["path"], "content": safe_path(args["path"]).read_text(encoding="utf-8", errors="replace")[:50000]}
    if name == "web_search":
        with httpx.Client(timeout=15, follow_redirects=True, headers={"User-Agent":"MECH-AI/1.0"}) as h:
            html = h.get("https://www.google.com/search", params={"q": args["query"]}).text
        text = " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split())
        return {"query": args["query"], "results": text[:8000]}
    if name == "remember":
        with db() as conn: conn.execute("INSERT OR REPLACE INTO memory VALUES (?, ?)", (args["key"], args["value"]))
        return {"saved": args["key"]}
    return {"error": "Unknown tool"}

def execute_approved(approval_id: str) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT action, arguments, used FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone()
        if not row: return {"error": "Approval not found or expired"}
        if row[2]: return {"error": "This approval was already used"}
        action, args = row[0], json.loads(row[1])
        conn.execute("UPDATE approvals SET used = 1 WHERE approval_id = ?", (approval_id,))
    if action == "write_file":
        p = safe_path(args["path"]); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(args["content"], encoding="utf-8"); return {"written": args["path"]}
    if action in {"run_command", "github_command"}:
        command = args["command"]
        if action == "github_command" and not command.strip().startswith("gh "): return {"error": "GitHub commands must start with gh"}
        r = subprocess.run(command, cwd=WORKSPACE, shell=True, capture_output=True, text=True, timeout=120)
        return {"returncode": r.returncode, "stdout": r.stdout[-12000:], "stderr": r.stderr[-12000:]}
    if action == "send_message":
        url = os.getenv("MESSAGE_WEBHOOK_URL")
        if not url: return {"error": "MESSAGE_WEBHOOK_URL is not configured"}
        r = httpx.post(url, json={"content": args["message"], "text": args["message"]}, timeout=15)
        return {"status_code": r.status_code, "response": r.text[:1000]}
    return {"error": "Unsupported approval"}

def chat(user_message: str) -> str:
    if not client: return "I’m online, but OPENAI_API_KEY is missing. Add it to your .env file and restart me."
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    with db() as conn: rows = conn.execute("SELECT role, content FROM chats ORDER BY id DESC LIMIT 12").fetchall()
    messages += [{"role": role, "content": content} for role, content in reversed(rows)]
    messages.append({"role": "user", "content": user_message})
    for _ in range(8):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0.25)
        msg = response.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or "I’m ready. What would you like me to do?"
            with db() as conn: conn.executemany("INSERT INTO chats(role, content) VALUES (?, ?)", [("user", user_message), ("assistant", answer)])
            return answer
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result_for(call.function.name, json.loads(call.function.arguments)))})
    return "I stopped after reaching my tool-call limit."

@app.get("/")
def home(): return FileResponse(BASE_DIR / "templates" / "index.html")

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    try: return {"response": chat(req.message)}
    except Exception as e: raise HTTPException(400, str(e))

@app.post("/api/approve")
def approve(req: ApprovalRequest):
    try: return {"response": json.dumps(execute_approved(req.approval_id), indent=2)}
    except Exception as e: raise HTTPException(400, str(e))

db().close()
