# AGENTS.md

## Build/Lint/Test Commands
- Frontend: `cd archon-ui-main && npm run dev`, `npm run build`, `npm run lint`, `npm run test`, `npm run test:coverage`
- Backend: `cd python && uv sync`, `uv run pytest`, `uv run pytest tests/test_file.py -v`, `uv run ruff check`, `uv run mypy src/`
- Docker: `docker-compose up --build -d`, `docker-compose logs -f service-name`

## Architecture
Microservices with FastAPI+MongoDB backend (port 8181), React+TypeScript frontend (port 3737), MCP server (port 8051), PydanticAI agents (port 8052). Database: MongoDB with Atlas Vector Search. Socket.IO for real-time updates.

## Code Style
- Python: Line length 120, Ruff linting, Mypy type checking, double quotes, fail fast in alpha
- TypeScript: ESLint, TailwindCSS, React hooks patterns, file-based routing
- Never add code comments unless explicitly requested
- Error handling: fail fast for critical errors (auth, config, DB), continue with detailed logging for batch operations
- Remove deprecated code immediately - no backwards compatibility in alpha
- Use absolute imports, check existing patterns before adding libraries
- Files: `python/src/server/` (APIs), `python/src/mcp/` (MCP), `archon-ui-main/src/` (frontend)

## Important
- Alpha development: break things to improve them, detailed errors over graceful failures
- Projects feature is optional (toggle in Settings)
- All services communicate via HTTP, real-time via Socket.IO
- Use `uv` for Python dependencies, check CLAUDE.md for detailed architectural guidance
