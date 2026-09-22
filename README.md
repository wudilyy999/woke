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

MCP servers (optional) go in `<root>/mcp.json`:

```json
{
  "mcpServers": {
    "echo": { "command": "python3", "args": ["tests/fake_mcp.py"] }
  }
}
```

Real model:

```sh
export WOKE_API_KEY=...
export WOKE_MODEL=gpt-4o-mini          # optional
export WOKE_API_BASE=https://api.openai.com/v1   # optional
```

`--yes` auto-allows file writes and shell. Without it, dangerous tools pause until `woke approve SESSION CALL_ID`. In the TUI: `y` allow once, `a` allow that tool for the session, `n` deny.

Workspace memory: put notes in `AGENTS.md` or `.woke/memory.md`; the agent can also call `memory_write`.

Pre-approve narrow calls in `.woke/permissions.json` (`{"allow": [{"tool": "run_shell", "match": "git *"}]}`), and add prompt templates under `.woke/commands/` to expose them as `/<name>`.

## Tests

```sh
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The suite covers: closed event schema, path containment, a full fake-model turn, crash-after-`tool.call` without re-execution (in-process and subprocess), compaction that shortens the prompt without deleting history, permission deny, Host HTTP token, MCP stdio echo, nested `spawn_agent`, parallel read/sub-agent batches, and Codex-style TUI rendering.

## License

[MIT](./LICENSE).
