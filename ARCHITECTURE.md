# woke architecture

woke is a small event-sourced coding-agent runtime. **The append-only event log is the only fact.** Session state, the next model prompt, the CLI view, and crash recovery are projections of that log.

The design is inspired by Apache Maka's published thesis ("the log is the runtime"). The code is an independent implementation, not a fork. Maka's Host Kernel, Domain modules, SessionManager, AgentRun, Graph, Peer Mesh, Computer Use, and Electron desktop are intentionally absent.

## Layers

```text
CLI  (thin HTTP client)
  → Host process  (admission, turn lock, sole writer)
      → Turn loop     (model + tools)
      → Event log     (SQLite, closed kinds, commit-then-side-effect)
      → Projection    (context / recovery / CLI)
```

One Host process owns one **state root** (a directory). That root may contain many sessions. At most one in-flight turn per session. A file lock makes the Host the only writer.

Write path:

```text
client command → Host admits → append log (durable) → tool side effect or projection
```

A mutation is not applied to the workspace and then recorded. The enabling event is committed first.

## State root

```text
<root>/
  root.json     # root_id, shared token
  host.json     # pid, port (ephemeral while Host is up)
  woke.lock     # exclusive writer lock
  woke.db       # append-only events
```

Default root: `~/.woke`.

## Event contract

Unknown `kind` values are rejected at the writer. v1 kinds:

| kind | role |
|---|---|
| `session.created` | workspace + title |
| `turn.started` / `turn.terminated` | one user task |
| `run.started` / `run.terminated` | one execution attempt (`fresh` or `recovery`) |
| `user.message` / `model.message` | conversation facts |
| `tool.call` / `tool.result` | tool facts |
| `permission.requested` / `permission.decided` | dangerous-tool gate |
| `compaction.applied` | LLM summary covering `from_seq`–`to_seq` |
| `todo.updated` | task list for the current turn, rendered by the TUI |

Envelope: `seq, ts, session_id, turn_id, run_id, kind, payload`.

`seq` is a global integer. Compaction never deletes rows.

`user.message` carries the text, the text attachments resolved from `@path` mentions, and `images` (workspace-relative paths). The prompt projection turns an image into an OpenAI `image_url` part with a `data:` URL, so the log stays readable and only the request grows.

## Recovery

On Host start, any turn with `turn.started` and no `turn.terminated` is recovered:

1. Every `tool.call` without a `tool.result` gets `error=interrupted_by_crash`. That tool is **not** executed again.
2. An open run is aborted. A new run is started with `reason=recovery` on the same turn.
3. Partial model output is not in the log (deltas are never stored), so inference is retried from the projected prompt.

This is the core test: kill the process after `tool.call`, restart, assert the file was not written, assert the interrupted result, assert the turn can finish.

Limitation: if the process dies after a side effect and before `tool.result`, woke still does not re-run the tool. The workspace may already contain that effect. Filesystem rollback is out of scope.

## Memory and compaction

Prompt construction:

1. System instructions.
2. Workspace briefing from `AGENTS.md`, `CLAUDE.md`, `.woke/instructions.md`, and `.woke/memory.md` (capped).
3. Latest `compaction.applied` summary, if any.
4. Events after that `to_seq`. Tool bodies are pruned in the prompt (8k chars); the log keeps the full output.
5. The latest `todo.updated` list, and the plan-mode instruction when the turn runs in plan mode.

`turn.started` records `mode`: `plan` or `execute`. A plan turn runs under the read-only policy, so dangerous tools are denied while the model inspects the workspace and writes a plan. `todo_write` appends `todo.updated`; the runner handles it so the list lands in the log rather than on disk.

`memory_read` / `memory_write` persist notes in `.woke/memory.md` so they survive compaction.

Compaction writes `compaction.applied` and never deletes events. A later compact **folds** the previous summary into the new one (`Previous summary:` + new chunk). It cuts only at `turn.terminated` or a completed `tool.result` so tool pairs stay aligned, and it repeats until the prompt is under budget (or four rounds). Summarizer failure is ignored; the turn continues.

Token estimate is `chars/4` (not a tokenizer). Default budget 32_000.

## Permissions

Dangerous: `write_file`, `str_replace`, `run_shell`, `spawn_agent`, `web_fetch`, `web_search`, non-readonly MCP. Reads, grep, `todo_write`, and `memory_write` are not gated.

Four levels (`/permission`):

| mode | writes | shell / spawn / MCP |
|---|---|---|
| `readonly` | deny | deny |
| `ask` | ask | ask |
| `edits` | auto-allow | ask |
| `auto` | auto-allow | auto-allow |

`--yes` starts in `auto`. While asking: `y` this call, `a` this tool for the session, `n` deny. Session grants still apply on top of the mode. Path containment always holds.

A workspace can pre-approve narrow calls in `.woke/permissions.json`:

```json
{"allow": [{"tool": "run_shell", "match": "git *"}]}
```

The pattern matches the tool's primary argument (`command` for `run_shell`, `path` for file tools, `url` for `web_fetch`, `query` for `web_search`), or the JSON-encoded arguments for any other tool. A rule turns `wait` into `allow`; `readonly` still denies and `auto` already allows. A malformed file fails the turn loudly instead of being ignored.

## Errors

Tool exceptions become `tool.result` failures; they do not crash the Host. Missing arguments, binary files, empty shell, and shell timeouts are structured errors. Model HTTP 429/5xx and transport errors retry twice, then the turn terminates `failed` with the error on the log. Host recovery of one session cannot take down the others.

## Host protocol

Loopback HTTP, shared token (`X-Woke-Token` or `Authorization: Bearer`). JSON request/response. Not a remote-access product: no TLS, no multi-machine.

`GET /search?q=` reads `user.message` and `model.message` bodies across every session in the root and returns one hit per session with a snippet and a match count. Child sessions are omitted. The CLI wraps it as `woke search`, and `/resume <text>` in the TUI merges those hits into the session picker.

`POST /sessions/{id}/turns` with `"background": true` returns `202` and runs the turn on a worker thread. `GET /sessions/{id}/events/stream?after=N` then delivers server-sent events as they are appended and closes on `turn.terminated`, or at once when there is nothing left to replay and no turn is open. A turn that dies outside the runner still writes a failed `run.terminated` / `turn.terminated` pair, so a follower never hangs. `HostClient.start_turn` plus `HostClient.stream_events` are the SDK for editors: import them from `woke.host`, or speak the two endpoints from any language.

## Tools

Workspace is a directory on the session, not a git worktree. Paths must stay inside it. Shell `cwd` is that directory. `run_shell` and MCP servers launch under a platform sandbox: macOS uses `sandbox-exec` with a seatbelt policy, Linux uses `bwrap`. Writes are confined to the workspace plus the tmp directories; reads and network access stay open, so `git`, `pip` and `npm` keep working. A shell call fails when neither sandbox binary is present. Search is stdlib, not ripgrep.

`run_shell` streams its output line by line to the caller and stops the process when the turn is cancelled. `web_fetch` returns the text of an http(s) URL; `web_search` queries DuckDuckGo lite and returns titles, links and snippets. Both network tools are dangerous, so they pass through the permission gate.

## Hooks

`.woke/hooks.json` holds `{"hooks": [{"event": ..., "match": ..., "command": ...}]}`. Events are `PreToolUse`, `PostToolUse`, and `TurnEnd`; snake_case spellings work too. The command runs through `sh` in the workspace with `cwd` set there and gets one JSON payload on stdin: `tool` and `arguments`, plus `ok` and the tail of the output for `PostToolUse`. A non-zero exit from `PreToolUse` blocks the tool and its stderr becomes the tool error; `PostToolUse` stdout is appended to the tool result; `TurnEnd` carries the final status, which is where completion notifications belong. Every run lands in the log as a `hook.result` event.

## MCP

The Host reads `<root>/mcp.json` (`mcpServers` map, same shape as Claude/Codex config). Each server is a stdio JSON-RPC process with LSP `Content-Length` framing. Tools are registered as `mcp__<server>__<tool>` and executed as ordinary `tool.call` / `tool.result` events. Read-only MCP tools skip the permission gate; others are dangerous.

An entry with `url` instead of `command` speaks streamable HTTP: one POST per JSON-RPC message, answered with either `application/json` or an `text/event-stream` frame.

Resources advertised by the server become a read-only `mcp__<server>__read_resource` tool whose description lists the available uris. Prompts advertised by the server become TUI commands `/<server>:<prompt>`; arguments are `name=value` pairs, and bare text fills a prompt that declares exactly one argument. `GET /mcp` exposes the lists over the Host API.

## Sub-agents

`spawn_agent` is a dangerous builtin. It creates a child session (`session.created.parent_session_id`) in the same workspace, runs one turn with auto-allow, and returns a digest. The child has its own log. Depth is capped at 1 (the child cannot spawn). This is a nested Engine, not a second Host.

Approved calls of one model reply are dispatched together. A batch made only of read-only tools and `spawn_agent` runs on a thread pool, so several sub-agents work at once; any batch containing a write or shell call keeps its order and runs one call at a time.

## TUI

`woke` with no subcommand opens a Codex-style terminal: welcome card, scrolling transcript, bottom `›` composer, cyan status line, magenta brand, slash commands (`/help` `/quit` `/clear` `/yes` `/compact`). Colors follow Codex's published TUI style notes. The TUI is a client of Host; it does not own the log.

## Out of scope

No Electron, peer mesh, computer-use, eval harness, or filesystem rollback. `replay <seq>` is still a natural later feature.
