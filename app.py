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
app = FastAPI(title="Local Coding Agent")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

class ChatRequest(BaseModel):
    message: str
    approved_action: str | None = None

class ApprovalRequest(BaseModel):
    approval_id: str

SYSTEM = """You are a careful local coding assistant. Help the user inspect and modify their project.
Use tools when useful. Always explain what you intend to do before dangerous actions.
Reading files and searching are safe. File writes, shell commands, GitHub commands, and sending
messages require confirmation: call the tool and return its approval_id instead of performing it.
Never expose API keys or secrets. Keep answers concise and include commands/results when relevant.
"""

TOOLS = [
    {"type":"function","function":{"name":"web_search","description":"Search the web and return concise page text.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"list_files","description":"List files under a workspace-relative directory.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a workspace-relative text file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Create or replace a workspace file. Requires user confirmation.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"run_command","description":"Run a command in the workspace. Requires user confirmation.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"github_command","description":"Run a read-only or mutating GitHub CLI command. Requires confirmation.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"send_message","description":"Send a message through the configured webhook. Requires confirmation.","parameters":{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}}},
    {"type":"function","function":{"name":"remember","description":"Save a user preference or project fact in local memory.","parameters":{"type":"object","properties":{"key":{"type":"string"},"value":{"type":"string"}},"required":["key","value"]}}},
]

DANGEROUS = {"write_file", "run_command", "github_command", "send_message"}

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, role TEXT, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    return conn

def safe_path(path: str) -> Path:
    candidate = (WORKSPACE / path).resolve()
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError("Path is outside the configured workspace")
    return candidate

def result_for(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name in DANGEROUS:
        return {"needs_approval": True, "action": name, "arguments": args, "approval_id": f"{name}:{abs(hash(json.dumps(args, sort_keys=True)))}"}
    if name == "list_files":
        p = safe_path(args["path"])
        return {"path": args["path"], "files": [str(x.relative_to(WORKSPACE)) for x in sorted(p.rglob("*")) if x.is_file()][:300]}
    if name == "read_file":
        p = safe_path(args["path"])
        return {"path": args["path"], "content": p.read_text(encoding="utf-8", errors="replace")[:50000]}
    if name == "web_search":
        with httpx.Client(timeout=15, follow_redirects=True, headers={"User-Agent":"LocalCodingAgent/1.0"}) as h:
            html = h.get("https://www.google.com/search", params={"q": args["query"]}).text
        soup = BeautifulSoup(html, "html.parser")
        text = " ".join(soup.get_text(" ").split())
        return {"query": args["query"], "results": text[:8000]}
    if name == "remember":
        with db() as c: c.execute("INSERT OR REPLACE INTO memory VALUES (?, ?)", (args["key"], args["value"]))
        return {"saved": args["key"]}
    return {"error": "Unknown tool"}

def execute_approved(action: str, args: dict[str, Any]) -> dict[str, Any]:
    if action == "write_file":
        p = safe_path(args["path"]); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(args["content"], encoding="utf-8"); return {"written": args["path"]}
    if action in {"run_command", "github_command"}:
        command = args["command"]
        if action == "github_command" and not command.strip().startswith("gh "): return {"error":"GitHub commands must start with gh"}
        r = subprocess.run(command, cwd=WORKSPACE, shell=True, capture_output=True, text=True, timeout=120)
        return {"returncode": r.returncode, "stdout": r.stdout[-12000:], "stderr": r.stderr[-12000:]}
    if action == "send_message":
        url = os.getenv("MESSAGE_WEBHOOK_URL")
        if not url: return {"error":"MESSAGE_WEBHOOK_URL is not configured"}
        r = httpx.post(url, json={"content": args["message"], "text": args["message"]}, timeout=15)
        return {"status_code": r.status_code, "response": r.text[:1000]}
    return {"error":"Unsupported approval"}

def chat(user_message: str, approved_action: str | None = None) -> str:
    if not client: return "Configure OPENAI_API_KEY in .env before using the agent."
    messages: list[dict[str, Any]] = [{"role":"system","content":SYSTEM}]
    with db() as c:
        rows = c.execute("SELECT role, content FROM chats ORDER BY id DESC LIMIT 12").fetchall()
    messages += [{"role": r, "content": c} for r, c in reversed(rows)]
    messages.append({"role":"user","content":user_message})
    if approved_action:
        messages.append({"role":"system","content":f"The user approved action {approved_action}. Execute the pending action only if it is in the previous approval request."})
    for _ in range(8):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0.2)
        msg = response.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or "Done."
            with db() as c: c.executemany("INSERT INTO chats(role, content) VALUES (?, ?)", [("user", user_message), ("assistant", answer)])
            return answer
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            tool_result = result_for(call.function.name, args)
            messages.append({"role":"tool","tool_call_id":call.id,"content":json.dumps(tool_result)})
    return "I stopped after reaching the tool-call limit."

@app.get("/")
def home(): return FileResponse(BASE_DIR / "templates" / "index.html")

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    try: return {"response": chat(req.message, req.approved_action)}
    except Exception as e: raise HTTPException(400, str(e))

@app.post("/api/approve")
def approve(req: ApprovalRequest):
    return {"response": "Approval received. Please resend the original request with approved_action set to the approval ID."}

db().close()
