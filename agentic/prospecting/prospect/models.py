"""Pydantic models for pipeline data."""

from pydantic import BaseModel


class Company(BaseModel):
    id: str
    name: str
    linkedin_url: str | None = None
    website_url: str | None = None
    size_range: str | None = None
    funding_stage: str | None = None
    industry: str | None = None
    summary: str | None = None
    industry_detected: str | None = None
    target_customer: str | None = None
    products_services: str | None = None
    tech_stack: str | None = None


class Person(BaseModel):
    id: str
    company_id: str
    name: str
    title: str
    linkedin_url: str | None = None
    email: str | None = None
    persona: str | None = None


class NewsItem(BaseModel):
    company_id: str
    headline: str
    url: str
    date: str | None = None
    snippet: str = ""
    full_summary: str | None = None


class DraftEmail(BaseModel):
    person_id: str
    company_id: str
    subject: str
    body: str
    persona_template: str
    version: int = 1
