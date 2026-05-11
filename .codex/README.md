# Codex App Configuration

This directory contains project-scoped Codex App settings for the KIK VLM MVP. Codex reads this file only when the project is trusted.

Trust the nested repository root itself: `/Users/semengolodnuk/Documents/ice_cream/ice-cream-vlm-mvp`. If the CLI says `config profile ... not found`, Codex is reading the parent workspace config instead of this project config.

## Profiles

`power` is the default profile because this repository is image-heavy and research-oriented. It enables live web search, high web-search context, detailed reasoning summaries, image viewing, more parallel subagents, and hooks support.

Run it explicitly:

```bash
codex --cd /Users/semengolodnuk/Documents/ice_cream/ice-cream-vlm-mvp --profile power
```

`daily` is the calmer profile for routine edits. It keeps cached web search, medium search context, concise summaries, image viewing, fewer subagents, and hooks disabled.

Run it explicitly:

```bash
codex --cd /Users/semengolodnuk/Documents/ice_cream/ice-cream-vlm-mvp --profile daily
```

## Included Flags

- `web_search = "live"` in `power`; `web_search = "cached"` in `daily`.
- `tools.web_search.context_size = "high"` in `power`; `"medium"` in `daily`.
- `tools_view_image = true` in both profiles.
- `model_reasoning_summary = "detailed"` in `power`; `"concise"` in `daily`.
- `model_verbosity = "high"` in `power`; `"medium"` in `daily`.
- `service_tier = "fast"` in both profiles.
- `review_model = "gpt-5.5"` in both profiles.
- `agents.max_threads = 12`, `agents.max_depth = 3`, and `agents.job_max_runtime_seconds = 3600` in `power`.
- `agents.max_threads = 6`, `agents.max_depth = 1`, and `agents.job_max_runtime_seconds = 1200` in `daily`.
- `features.codex_hooks = true` only in `power`.

Current Codex resolves relative paths in project config from the `.codex/` directory, so `model_instructions_file = "../AGENTS.md"` points at the repository-level instructions file.

## Tuning Agents

Edit `.codex/config.toml`:

```toml
[profiles.power.agents]
max_threads = 12
max_depth = 3
job_max_runtime_seconds = 3600
```

Lower `max_threads` if local resource use gets noisy. Keep `max_depth = 1` unless recursive decomposition is genuinely useful.

## Permissions

The config defines three filesystem permission profiles:

- `safe`: read-only project access, with `.env` denied.
- `workspace`: write access inside project roots, with `.env` denied.
- `power`: write access inside project roots, with `.env` denied.

No broad network, Unix socket, proxy, or destructive app-tool permissions are enabled by default.

## Hooks

Hooks support is enabled only in the `power` profile. No active project hook is wired yet; `.codex/hooks/` is reserved for safe, explicit lifecycle scripts.

Disable hooks:

```toml
[profiles.power.features]
codex_hooks = false
```

Avoid hooks that commit, push, delete files, install dependencies, or mutate the environment without explicit confirmation.

## Web Search

Disable live web search by changing the active profile:

```toml
[profiles.power]
web_search = "cached"
```

Use `"disabled"` when a task should not use web search at all.

## MCP

No project-local MCP servers were found, so `.codex/config.toml` does not define active MCP servers. Locally observed user-level MCP servers were `fetch`, `sequential-thinking`, `context7`, and `github`; they remain in the user config, not this repo.

If a future project-critical MCP server is added, configure it explicitly:

```toml
[mcp_servers.example]
enabled = true
required = true
startup_timeout_ms = 30000
tool_timeout_sec = 300
# command = "example-mcp-server"
# args = []
# or:
# url = "https://example.com/mcp"
```

For non-critical MCP servers, use `required = false` and usually `tool_timeout_sec = 180`. Do not store bearer tokens or API keys directly in this file; use environment variables such as `bearer_token_env_var`.

## Compatibility Notes

This config was checked against local `codex-cli 0.128.0-alpha.1` and the current OpenAI Codex configuration docs. In current syntax, profile-local image support uses `tools_view_image = true`; older examples may show nested `tools.view_image`.
