# woke

A small **event-sourced coding-agent runtime**. The append-only log is the only fact; the next prompt, the CLI, and crash recovery are projections of that log.

Inspired by [Apache Maka](https://github.com/apache/maka)'s published "log is the runtime" thesis. See [ARCHITECTURE.md](./ARCHITECTURE.md) for the runtime contract.

## Requirements

- Python 3.11+ (developed against 3.12)
- No runtime dependencies. `sqlite3`, `http.server`, and `urllib` are stdlib.

## Quick start

Interactive TUI (Codex-style composer and transcript):

```sh
cd /path/to/your-project
woke
```

Workspace is the directory you launched from. No `--workspace` needed.

Model config is read from `~/.woke/config.toml` if present, else `~/.kimi-code/config.toml`. Keys stay in those home files; they are never written into this repo. See `config.example.toml`. `WOKE_API_KEY` / `WOKE_API_BASE` override. `/model` switches. `woke models` lists ids only.

Logs live under `~/.woke/ws/<hash>/` per workspace so two projects can run at once. `/rewind` (or Esc Esc) forks an earlier user turn; `/fork` copies the session. Workspace files are not reverted.

Enter sends, `y`/`n` answer permission prompts, `/plan` toggles read-only planning turns, `/help` lists slash commands, `/quit` exits.

Screenshots go in with `/image PATH` (queued for the next message) or `@path.png` in the text; the CLI takes `woke send --image PATH`. Images are sent as `image_url` parts, so the model needs vision support.

Headless Host + CLI still work:

```sh
python3 -m woke --root /tmp/woke-demo host start --detach --yes
python3 -m woke --root /tmp/woke-demo session new --workspace /tmp/woke-ws
python3 -m woke --root /tmp/woke-demo send <session-id> "hello" --yes
python3 -m woke --root /tmp/woke-demo events <session-id>
python3 -m woke --root /tmp/woke-demo search "quota"
python3 -m woke --root /tmp/woke-demo host stop
```

`search` reads every session transcript in the root. `/resume <text>` in the TUI searches transcripts too, so you can find an older session by something the agent said rather than by its title.

Editors and scripts drive the same Host: `woke send --background SESSION "..."` returns immediately, and `woke events SESSION --follow` prints the turn as it happens. In Python, `from woke.host import HostClient` gives you `start_turn`, `stream_events`, `decide_permission`, `cancel_turn` and `search_sessions`; every other language can POST `/sessions/{id}/turns` with `"background": true` and read `GET /sessions/{id}/events/stream` as server-sent events.

MCP servers (optional) go in `<root>/mcp.json`:

```json
{
  "mcpServers": {
    "echo": { "command": "python3", "args": ["tests/fake_mcp.py"] },
    "remote": { "url": "http://127.0.0.1:8787/mcp" }
  }
}
```

Servers may be stdio processes (`command`) or streamable HTTP endpoints (`url`). Their resources are readable through `mcp__<server>__read_resource`, and their prompts show up as `/<server>:<prompt>` in the TUI.

Real model:

```sh
export WOKE_API_KEY=...
export WOKE_MODEL=gpt-4o-mini          # optional
export WOKE_API_BASE=https://api.openai.com/v1   # optional
```

`--yes` auto-allows file writes and shell. Without it, dangerous tools pause until `woke approve SESSION CALL_ID`. In the TUI: `y` allow once, `a` allow that tool for the session, `n` deny.

Workspace memory: put notes in `AGENTS.md` or `.woke/memory.md`; the agent can also call `memory_write`.

Pre-approve narrow calls in `.woke/permissions.json` (`{"allow": [{"tool": "run_shell", "match": "git *"}]}`), and add prompt templates under `.woke/commands/` to expose them as `/<name>`.

Hooks go in `.woke/hooks.json`. `PreToolUse` can veto a call, `PostToolUse` can add feedback to the tool result, and `TurnEnd` runs when the turn stops:

```json
{
  "hooks": [
    { "event": "PreToolUse", "match": "run_shell", "command": "if grep -q 'rm -rf'; then echo 'rm -rf is not allowed here' >&2; exit 2; fi" },
    { "event": "PostToolUse", "match": "write_file", "command": "npx prettier --write \"$(jq -r .arguments.path)\"" },
    { "event": "TurnEnd", "command": "osascript -e 'display notification \"turn finished\"'" }
  ]
}
```

Each hook receives the call as JSON on stdin and runs in the workspace.

## Tests

```sh
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The suite covers: closed event schema, path containment, a full fake-model turn, crash-after-`tool.call` without re-execution (in-process and subprocess), compaction that shortens the prompt without deleting history, permission deny, Host HTTP token, MCP stdio echo, MCP resources/prompts over stdio and streamable HTTP, nested `spawn_agent`, parallel read/sub-agent batches, image attachments, cross-session search, hooks, and Codex-style TUI rendering.

## License

[MIT](./LICENSE).
