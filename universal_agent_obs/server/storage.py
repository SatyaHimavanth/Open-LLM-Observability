import os
import time
from typing import Optional

from sqlalchemy import JSON, Boolean, Float, String, create_engine, delete, func, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.getenv("AGENT_OBS_DB_URL", "sqlite:///./agent_obs.sqlite3")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class SpanRecord(Base):
    __tablename__ = "spans"

    span_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    framework: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    project_name: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    user_email: Mapped[Optional[str]] = mapped_column(String(256), index=True)
    resource: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    event: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    ts_start: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    received_at: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    archived: Mapped[bool] = mapped_column(Boolean, index=True, default=False, nullable=False)
    archived_at: Mapped[Optional[float]] = mapped_column(Float, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)

class UserRecord(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password: Mapped[str] = mapped_column(String(128))
    client_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    client_secret: Mapped[str] = mapped_column(String(128))

class ProjectRecord(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


def init_db():
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()


def upsert_span(session: Session, span: dict):
    user = span.get("user") or {}
    session.merge(
        SpanRecord(
            span_id=span["span_id"],
            trace_id=span["trace_id"],
            framework=span.get("framework"),
            project_name=span.get("project_name"),
            user_id=span.get("system_user_id"),
            user_email=user.get("email"),
            resource=span.get("resource"),
            event=span.get("event"),
            ts_start=span.get("ts_start") or 0,
            received_at=span.get("received_at") or 0,
            archived=bool(span.get("archived", False)),
            archived_at=span.get("archived_at"),
            payload=span,
        )
    )
    session.commit()


def get_trace_spans(session: Session, trace_id: str, include_archived: bool = False) -> list[dict]:
    stmt = select(SpanRecord).where(SpanRecord.trace_id == trace_id)
    if not include_archived:
        stmt = stmt.where(SpanRecord.archived.is_(False))
    rows = session.scalars(
        stmt.order_by(SpanRecord.ts_start.asc(), SpanRecord.received_at.asc())
    ).all()
    return [_payload_from_record(r) for r in rows]


def list_span_payloads(
    session: Session,
    trace_id: Optional[str] = None,
    limit: int = 500,
    include_archived: bool = False,
) -> list[dict]:
    stmt = select(SpanRecord)
    if not include_archived:
        stmt = stmt.where(SpanRecord.archived.is_(False))
    if trace_id:
        stmt = stmt.where(SpanRecord.trace_id == trace_id)
    rows = session.scalars(
        stmt.order_by(SpanRecord.ts_start.desc(), SpanRecord.received_at.desc()).limit(limit)
    ).all()
    return [_payload_from_record(r) for r in rows]


def list_trace_ids(
    session: Session,
    limit: int = 100,
    framework: Optional[str] = None,
    project: Optional[str] = None,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None,
    include_archived: bool = False,
) -> list[str]:
    stmt = select(SpanRecord.trace_id)
    if not include_archived:
        stmt = stmt.where(SpanRecord.archived.is_(False))
    if framework:
        stmt = stmt.where(SpanRecord.framework == framework)
    if project:
        stmt = stmt.where(SpanRecord.project_name == project)
    if user_email:
        stmt = stmt.where(func.lower(SpanRecord.user_email) == user_email.lower())
    if user_id:
        stmt = stmt.where(SpanRecord.user_id == user_id)
    rows = session.execute(
        stmt.group_by(SpanRecord.trace_id)
        .order_by(func.max(SpanRecord.received_at).desc())
        .limit(limit)
    ).all()
    return [r[0] for r in rows]


def archive_trace(session: Session, trace_id: str) -> int:
    archived_at = time.time()
    result = session.execute(
        update(SpanRecord)
        .where(SpanRecord.trace_id == trace_id)
        .where(SpanRecord.archived.is_(False))
        .values(archived=True, archived_at=archived_at)
    )
    session.commit()
    return result.rowcount or 0


def archive_project(session: Session, project_name: str) -> int:
    archived_at = time.time()
    result = session.execute(
        update(SpanRecord)
        .where(SpanRecord.project_name == project_name)
        .where(SpanRecord.archived.is_(False))
        .values(archived=True, archived_at=archived_at)
    )
    session.commit()
    return result.rowcount or 0


def archive_all(session: Session) -> int:
    archived_at = time.time()
    result = session.execute(
        update(SpanRecord)
        .where(SpanRecord.archived.is_(False))
        .values(archived=True, archived_at=archived_at)
    )
    session.commit()
    return result.rowcount or 0


def clear_all(session: Session):
    session.execute(delete(SpanRecord))
    session.commit()


def _payload_from_record(record: SpanRecord) -> dict:
    payload = dict(record.payload)
    payload["archived"] = record.archived
    if record.archived_at is not None:
        payload["archived_at"] = record.archived_at
    return payload
