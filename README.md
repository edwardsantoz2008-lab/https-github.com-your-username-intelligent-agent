# Local Coding Agent

A downloadable local web app powered by the OpenAI API. It can search the web, inspect and edit files, remember project facts, run local commands, use the GitHub CLI, and send webhook messages.

## Run locally

1. Install Python 3.10+.
2. Download or clone this repository.
3. Create a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
```

4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
6. Start it: `python app.py`
7. Open http://127.0.0.1:8000

Set `WORKSPACE_DIR` to the project the agent should manage. By default it is this repository. Set `MESSAGE_WEBHOOK_URL` only if you want the message tool enabled. GitHub features require the GitHub CLI (`gh`) and `gh auth login`.

## Safety

File writes, shell commands, GitHub commands, and outgoing messages are designed to require confirmation before execution. Do not run this app against a sensitive directory without reviewing commands first. Never commit `.env` or API keys.

## Example prompts

- Read `README.md` and explain this project.
- Search the web for the latest FastAPI testing guidance.
- List the Python files and find the API entry point.
- Run the tests and summarize failures.
- Create a test for the login function (the app will request confirmation before writing).
