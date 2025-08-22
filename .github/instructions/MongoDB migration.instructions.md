MongoDB migration.

What I checked

Env files: ".env.example", "docker-compose.yml"
Backend deps: "pyproject.toml"
Frontend deps: "archon-ui-main/package.json"
Code search for “supabase” and “mongodb” across backend
Results summary

Env mismatch:

.env template is MongoDB-only. Good.
docker-compose still exports SUPABASE_URL and SUPABASE_SERVICE_KEY to all Python services. This conflicts with MongoDB config and needs replacement.
Backend code state:

MongoDB config is present and healthy: "python/src/server/config/mongodb_config.py". It exposes get_mongodb_database() returning AsyncIOMotorDatabase.
A compatibility alias maps get_supabase_client → get_mongodb_client in "python/src/server/utils/init.py".
But core services still use the Supabase-style API (client.table(...).select().eq().execute()):
Projects: "project_service.py"
Tasks: "task_service.py"
Document/versioning/source-linking services under "python/src/server/services/projects"
Prompt and RAG wrappers: "prompt_service.py", "search/rag_service.py"
APIs: "projects_api.py" calls those services.
There are MongoDB-native services for knowledge, RAG, storage, credentials (good):
e.g., "mongodb_knowledge_services.py", "mongodb_rag_service.py", "mongodb_document_storage.py", "mongodb_credential_service.py"
Lint/type/test runs:

Ruff: 737 issues, mostly formatting and a handful of real problems (e.g., undefined names, unused variables). Example: "mongodb_search_strategies.py".
Mypy: 585 errors. The most serious are:
The Supabase-style API is still used everywhere, but get_supabase_client() now returns AsyncIOMotorDatabase. Mypy flags hundreds of “object has no attribute table/eq/execute” across project/task/versioning/document services. Example: "project_service.py#L53".
Optional defaults under strict no_implicit_optional=True need | None or non-None defaults.
Some API schemas/logging calls have incorrect types/kwargs.
Config dataclass ordering bug in "config.py": non-default field after default.
Pytest: collection fails immediately due to:
Removed/renamed symbols in credential service tests (import mismatch) at "tests/test_async_credential_service.py".
Dataclass arg ordering bug in "config.py" via "tests/test_supabase_validation.py".
Frontend build timed out; no MongoDB references found in app code beyond docs/strings. The frontend should be fine once the backend APIs run.
Dependencies:

Python still lists Postgres bits: asyncpg in "pyproject.toml". Likely unused now.
MCP minimal requirements still include supabase in "python/requirements.mcp.txt".
Repo still contains legacy “original_archon” Supabase code; fine to keep as history, but it will confuse searches/tests unless isolated.
What to fix (in order)

Environment and compose
Replace Supabase envs with Mongo in "docker-compose.yml":
Remove SUPABASE_URL, SUPABASE_SERVICE_KEY from archon-server, archon-mcp, archon-agents.
Add:
MONGODB_CONNECTION_STRING
MONGODB_DATABASE=archon (or your DB name)
Example for archon-server:
Before:
SUPABASE_URL=${SUPABASE_URL}
SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
After:
MONGODB_CONNECTION_STRING=${MONGODB_CONNECTION_STRING}
MONGODB_DATABASE=${MONGODB_DATABASE:-archon}
Decide data-access strategy for migrated modules
You have two options to remove the Supabase API surface from runtime:
Option A: Implement a thin Supabase-like adapter for Mongo

Provide a small adapter that exposes table(name) returning an object with chainable select()/eq()/in_()/order()/single()/update()/insert()/delete()/execute() and translate to Motor queries/aggregations.
Scope to tables used by “projects/tasks/sources/versions” paths.
Pros: Lowest surface change; keeps service code mostly intact.
Cons: Needs careful semantics mapping (e.g., .single(), .count="exact", filters like .or_("archived.is.null,archived.is.false"), JSONB-ish fields).
Option B: Refactor project/task/versioning/document services to use Motor directly

Update "project_service.py", "task_service.py", "document_service.py", "versioning_service.py", "source_linking_service.py" to use db.collection with Motor’s find_one, insert_one, update_one, delete_one, and projections/sorts.
This is the clean long-term path and aligns with existing MongoDB-native modules you already built.
Given the number of MyPy errors and API differences, I recommend Option B. You already have good Mongo patterns in:

"mongodb_knowledge_services.py"
"mongodb_storage_services.py"
Use those as reference.
Fix APIRouters to use Mongo-backed services consistently
"projects_api.py" and related APIs currently import Supabase-centric services but should call your Mongo versions once refactored.
Remove get_supabase_client() usages and instead call get_mongodb_database() directly where needed or pass db via service constructors.
Stabilize credentials API with Mongo service
Tests fail importing legacy names from "credential_service.py". That file now re-exports Mongo implementation names; update test imports or re-add aliases expected by tests, or mark those Supabase tests as skipped.
Type and dataclass fixes to unblock tests
Fix dataclass field ordering in "config.py" (non-default args must come before defaults).
Address easy mypy errors:
Add | None where defaults are None.
Fix logger calls that pass unknown kwargs.
Add missing annotations flagged by mypy.
Run: ruff --fix; this auto-resolves a few hundred style issues.
Remove stale dependencies
Remove asyncpg from "pyproject.toml" if no Postgres remains.
Consider removing supabase from "python/requirements.mcp.txt" if MCP features don’t require it any more.
Concrete hotspots to refactor first

Projects/tasks stack:
"project_service.py"
"task_service.py"
"projects_api.py"
Prompt/knowledge bits still referencing Supabase interface:
"prompt_service.py"
"knowledge_item_service.py" if used
Remove (or adapt) Supabase-heavy tests:
"tests/test_supabase_validation.py"
"tests/test_rag_simple.py", "tests/test_rag_strategies.py" reference Supabase mocks.
Suggested next steps (I can implement if you want)

Update docker-compose envs to Mongo.
Refactor ProjectService and TaskService to Motor (create collections: archon_projects, archon_tasks, archon_project_sources).
Adjust projects_api to drop get_supabase_client and rely on Mongo-backed services.
Remove asyncpg, drop Supabase from MCP requirements if unused.
Run ruff --fix; fix mypy blockers (dataclass order, Optionals).
Mark legacy Supabase tests as skipped or port them to Mongo.
Notes

The backward-compat alias get_supabase_client = get_mongodb_client is not sufficient; Motor’s AsyncIOMotorDatabase does not expose Supabase’s query-builder methods, which explains most type and runtime issues.
Frontend looks clean; once backend endpoints are wired to Mongo services, npm run build should pass.
If you want, I can start by:

Patching docker-compose envs.
Refactoring "project_service.py" to Motor and updating "projects_api.py". This will eliminate hundreds of mypy errors in one shot and make the core flows work with MongoDB.