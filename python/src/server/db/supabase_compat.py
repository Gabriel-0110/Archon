"""
Supabase-compatible query adapter over MongoDB (synchronous) to ease migration.

This provides a minimal subset of the Supabase Python client surface used by the
current codebase: table().select().eq()/neq()/in_()/gte()/order().single().execute(),
plus insert/update/delete flows that return an object with a `.data` attribute.

Implementation uses the synchronous PyMongo client so it can be called from
non-async service code without event loop issues.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from ..config.mongodb_config import get_mongodb_config, get_sync_mongodb_client


@dataclass
class _Result:
    data: Any
    count: int | None = None


class _TableQuery:
    def __init__(self, collection_name: str, db):
        self._db = db
        self._collection = db[collection_name]
        self._collection_name = collection_name

        # operation state
        self._op: str | None = None  # "select" | "insert" | "update" | "delete"
        self._insert_payload: Any | None = None
        self._update_payload: dict[str, Any] | None = None
        self._single: bool = False
        self._count_mode: str | None = None

        # query state
        self._filters: dict[str, Any] = {}
        self._or_filters: list[dict[str, Any]] | None = None
        self._projection: dict[str, int] | None = None
        self._sort: list[tuple[str, int]] = []

    # ----- Supabase-like API -----
    def select(self, fields: str = "*", count: str | None = None) -> _TableQuery:
        self._op = "select"
        self._count_mode = count
        if fields and fields.strip() != "*":
            proj: dict[str, int] = {}
            for f in [p.strip() for p in fields.split(",") if p.strip()]:
                proj[f] = 1
            # Always include id if stored
            proj.setdefault("id", 1)
            self._projection = proj
        else:
            self._projection = None
        return self

    def single(self) -> _TableQuery:
        self._single = True
        return self

    def eq(self, field: str, value: Any) -> _TableQuery:
        self._add_filter(field, value)
        return self

    def neq(self, field: str, value: Any) -> _TableQuery:
        self._add_filter(field, {"$ne": value})
        return self

    def gte(self, field: str, value: Any) -> _TableQuery:
        self._add_filter(field, {"$gte": value})
        return self

    def in_(self, field: str, values: list[Any]) -> _TableQuery:
        self._add_filter(field, {"$in": list(values)})
        return self

    def or_(self, expr: str) -> _TableQuery:
        # Only implement the pattern used in code: archived.is.null,archived.is.false
        if expr.strip() == "archived.is.null,archived.is.false":
            clause: list[dict[str, Any]] = [{"archived": {"$exists": False}}, {"archived": False}]
            if self._or_filters is None:
                self._or_filters = clause
            else:
                self._or_filters.extend(clause)
        return self

    def order(self, field: str, desc: bool = False) -> _TableQuery:
        self._sort.append((field, -1 if desc else 1))
        return self

    def insert(self, payload: Any) -> _TableQuery:
        self._op = "insert"
        self._insert_payload = payload
        return self

    def update(self, payload: dict[str, Any]) -> _TableQuery:
        self._op = "update"
        self._update_payload = payload
        return self

    def delete(self) -> _TableQuery:
        self._op = "delete"
        return self

    # ----- Execution -----
    def execute(self) -> _Result:
        if self._op == "insert":
            return self._exec_insert()
        if self._op == "update":
            return self._exec_update()
        if self._op == "delete":
            return self._exec_delete()
        # default to select
        return self._exec_select()

    # ----- Internals -----
    def _build_filter(self) -> dict[str, Any]:
        flt: dict[str, Any] = dict(self._filters)
        if self._or_filters:
            flt = {"$and": [flt, {"$or": self._or_filters}]} if flt else {"$or": self._or_filters}
        return flt

    def _add_filter(self, field: str, condition: Any) -> None:
        # NOTE: only dot-path fields and direct names supported
        if "$and" in self._filters or "$or" in self._filters:
            # shouldn't happen with our builder, but keep safe
            base = self._filters
        else:
            base = self._filters
        if field in base and isinstance(base[field], dict) and isinstance(condition, dict):
            # merge operators
            base[field].update(condition)
        else:
            base[field] = condition

    def _ensure_ids_on_insert(self, doc: dict[str, Any]) -> dict[str, Any]:
        if "id" not in doc:
            doc["id"] = uuid.uuid4().hex
        return doc

    def _exec_insert(self) -> _Result:
        assert self._insert_payload is not None
        payload = self._insert_payload
        if isinstance(payload, list):
            docs = [self._ensure_ids_on_insert(dict(d)) for d in payload]
            res = self._collection.insert_many(docs)
            # fetch inserted docs by their ids
            ids = [d["id"] for d in docs]
            out = list(self._collection.find({"id": {"$in": ids}}, projection=self._projection))
            return _Result(self._normalize_docs(out), count=len(out))
        else:
            doc = self._ensure_ids_on_insert(dict(payload))
            self._collection.insert_one(doc)
            out = self._collection.find_one({"id": doc["id"]}, projection=self._projection)
            return _Result(self._normalize_doc(out), count=1)

    def _exec_update(self) -> _Result:
        assert self._update_payload is not None
        flt = self._build_filter()
        self._collection.update_many(flt, {"$set": dict(self._update_payload)})
        # Return updated docs
        cursor = self._collection.find(flt, projection=self._projection)
        if self._sort:
            cursor = cursor.sort(self._sort)
        docs = list(cursor)
        if self._single:
            return _Result(self._normalize_doc(docs[0] if docs else None), count=1 if docs else 0)
        return _Result(self._normalize_docs(docs), count=len(docs))

    def _exec_delete(self) -> _Result:
        flt = self._build_filter()
        self._collection.delete_many(flt)
        return _Result([])

    def _exec_select(self) -> _Result:
        flt = self._build_filter()
        if self._single:
            doc = self._collection.find_one(flt, projection=self._projection)
            return _Result(self._normalize_doc(doc), count=1 if doc else 0)
        cursor = self._collection.find(flt, projection=self._projection)
        if self._sort:
            cursor = cursor.sort(self._sort)
        docs = list(cursor)
        return _Result(self._normalize_docs(docs), count=len(docs) if self._count_mode else None)

    def _normalize_doc(self, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if not doc:
            return None
        d = dict(doc)
        d.pop("_id", None)
        return d

    def _normalize_docs(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in docs:
            nd = self._normalize_doc(d)
            if nd is not None:
                out.append(nd)
        return out


class SupabaseCompat:
    """Entry point mimicking the Supabase client with a `table` method."""

    def __init__(self):
        cfg = get_mongodb_config()
        client = get_sync_mongodb_client()
        self._db = client[cfg.database_name]

    def table(self, name: str) -> _TableQuery:
        return _TableQuery(name, self._db)

    # Optional convenience for future: RPC placeholder to avoid crashes if called
    def rpc(self, *_args, **_kwargs):  # pragma: no cover - not used in migrated paths
        class _RPC:
            def execute(self):
                return _Result([])
        return _RPC()
