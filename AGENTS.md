# AGENTS.md

## Repository Expectations

- Start by reading `README.md` and inspecting the current project structure before changing files.
- Before edits, state a brief plan for non-trivial work.
- Keep changes scoped. Do not rewrite large files wholesale unless the task truly requires it.
- Do not break public CLI flags, output files, schemas, or report formats without a clear reason.
- Preserve existing user changes in the worktree. Do not revert unrelated edits.
- Keep secrets out of commits and logs. Do not print `.env` contents or API keys.
- After changes, run the relevant tests or linters when they exist. For this repo, prefer `python3 -m unittest discover -s tests -v` for code changes.
- If tests or linters cannot run, report the exact blocker.
- For research tasks, prefer live web search and cite current sources.
- For image, screenshot, or photo tasks, use the local image viewing tool before making visual claims.
- For large tasks, split work into bounded sub-tasks and use subagents where that materially improves throughput.
