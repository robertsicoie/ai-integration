"""Database layer using SQLAlchemy Core with SQLite."""

from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import create_engine

from prospect.config import settings

metadata = sa.MetaData()

companies = sa.Table(
    "companies",
    metadata,
    sa.Column("id", sa.String, primary_key=True),
    sa.Column("name", sa.String, nullable=False),
    sa.Column("linkedin_url", sa.String),
    sa.Column("website_url", sa.String),
    sa.Column("size_range", sa.String),
    sa.Column("funding_stage", sa.String),
    sa.Column("industry", sa.String),
    sa.Column("summary", sa.Text),
    sa.Column("industry_detected", sa.String),
    sa.Column("target_customer", sa.String),
    sa.Column("products_services", sa.Text),
    sa.Column("tech_stack", sa.Text),
    sa.Column("scrape_status", sa.String, default="pending"),
    sa.Column("research_status", sa.String, default="pending"),
    sa.Column("news_status", sa.String, default="pending"),
    sa.Column("created_at", sa.DateTime, default=lambda: datetime.now(timezone.utc)),
    sa.Column("updated_at", sa.DateTime, default=lambda: datetime.now(timezone.utc)),
)

persons = sa.Table(
    "persons",
    metadata,
    sa.Column("id", sa.String, primary_key=True),
    sa.Column("company_id", sa.String, sa.ForeignKey("companies.id")),
    sa.Column("name", sa.String, nullable=False),
    sa.Column("title", sa.String),
    sa.Column("linkedin_url", sa.String),
    sa.Column("email", sa.String),
    sa.Column("persona", sa.String),
    sa.Column("created_at", sa.DateTime, default=lambda: datetime.now(timezone.utc)),
)

news_items = sa.Table(
    "news_items",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("company_id", sa.String, sa.ForeignKey("companies.id")),
    sa.Column("headline", sa.String),
    sa.Column("url", sa.String),
    sa.Column("date", sa.String),
    sa.Column("snippet", sa.Text),
    sa.Column("full_summary", sa.Text),
    sa.Column("created_at", sa.DateTime, default=lambda: datetime.now(timezone.utc)),
)

draft_emails = sa.Table(
    "draft_emails",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("person_id", sa.String, sa.ForeignKey("persons.id")),
    sa.Column("company_id", sa.String, sa.ForeignKey("companies.id")),
    sa.Column("subject", sa.String),
    sa.Column("body", sa.Text),
    sa.Column("persona_template", sa.String),
    sa.Column("version", sa.Integer, default=1),
    sa.Column("created_at", sa.DateTime, default=lambda: datetime.now(timezone.utc)),
)

rate_limits = sa.Table(
    "rate_limits",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("action_type", sa.String, nullable=False),
    sa.Column("count", sa.Integer, default=0),
    sa.Column("window_start", sa.DateTime),
)


def get_engine():
    return create_engine(
        f"sqlite:///{settings.db_path}",
        echo=False,
        connect_args={"timeout": 30},
    )


def init_db():
    """Create all tables if they don't exist."""
    engine = get_engine()
    metadata.create_all(engine)
    return engine


def get_connection():
    engine = init_db()
    return engine.connect()
