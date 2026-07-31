"""Stage 1 (alt): Apollo.io API-based prospecting.

Alternative to LinkedIn Sales Navigator scraping.
Uses Apollo's search API to find companies and decision makers matching ICP.
"""

import hashlib
import re
import time

import httpx
import sqlalchemy as sa
from rich.console import Console
from rich.progress import Progress

from prospect.config import settings
from prospect.db import companies, persons, init_db

console = Console()

APOLLO_BASE = "https://api.apollo.io/api/v1"

# Map ICP company sizes to Apollo's num_employees_ranges
SIZE_MAP = {
    "1-10": "1,10",
    "11-50": "11,50",
    "51-200": "51,200",
    "201-500": "201,500",
    "501-1000": "501,1000",
}


def _make_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    short_hash = hashlib.md5(name.encode()).hexdigest()[:6]
    return f"{slug}-{short_hash}"


def _map_persona(title: str) -> str:
    title_lower = title.lower()
    if any(t in title_lower for t in ["ceo", "chief executive", "founder", "co-founder"]):
        return "founder_ceo"
    if any(t in title_lower for t in ["cto", "chief technology", "chief tech"]):
        return "cto"
    if any(t in title_lower for t in ["vp eng", "vice president eng", "head of eng", "director of eng"]):
        return "vp_engineering"
    if any(t in title_lower for t in ["head of product", "product director", "vp product"]):
        return "head_of_product"
    return "other"


def _apollo_headers() -> dict:
    return {
        "x-api-key": settings.apollo_api_key,
        "Content-Type": "application/json",
    }


def _build_employee_ranges() -> list[str]:
    """Convert ICP company sizes to Apollo's format."""
    ranges = []
    for size in settings.get_company_sizes():
        if size in SIZE_MAP:
            ranges.append(SIZE_MAP[size])
        else:
            # Try to parse "X-Y" directly
            match = re.match(r"(\d+)-(\d+)", size)
            if match:
                ranges.append(f"{match.group(1)},{match.group(2)}")
    return ranges


def _search_companies(page: int = 1, per_page: int = 25) -> dict:
    """Search Apollo for companies matching ICP filters."""
    payload = {
        "page": page,
        "per_page": per_page,
    }

    # Employee count ranges
    ranges = _build_employee_ranges()
    if ranges:
        payload["num_employees_ranges"] = ranges

    # Industries (Apollo uses free-text industry keywords)
    industries = settings.get_industries()
    if industries:
        payload["industry_tag_ids"] = []
        payload["q_organization_keyword_tags"] = industries

    # Locations
    locations = settings.get_locations()
    if locations:
        payload["organization_locations"] = locations

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{APOLLO_BASE}/mixed_companies/search",
            headers=_apollo_headers(),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


def _search_people(
    organization_ids: list[str] | None = None,
    titles: list[str] | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict:
    """Search Apollo for people matching filters."""
    payload = {
        "page": page,
        "per_page": per_page,
    }

    if organization_ids:
        payload["organization_ids"] = organization_ids

    if titles:
        payload["person_titles"] = titles

    # Seniority filter for decision makers
    payload["person_seniorities"] = ["c_suite", "vp", "director", "founder"]

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{APOLLO_BASE}/mixed_people/search",
            headers=_apollo_headers(),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def scrape_apollo(max_results: int = 200):
    """Main entry point for Apollo.io prospecting."""
    if not settings.apollo_api_key:
        console.print("[red]APOLLO_API_KEY not set in .env[/red]")
        return

    engine = init_db()

    console.print("[bold]Stage 1: Apollo.io Prospecting[/bold]")

    # ── Phase 1: Search for companies ──
    console.print("\n[bold]Phase 1: Searching for companies...[/bold]")

    total_companies = 0
    page = 1
    per_page = min(25, max_results)

    with Progress() as progress:
        task = progress.add_task("Searching companies...", total=max_results)

        while total_companies < max_results:
            try:
                result = _search_companies(page=page, per_page=per_page)
            except httpx.HTTPStatusError as e:
                console.print(f"[red]Apollo API error: {e.response.status_code} — {e.response.text[:200]}[/red]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                break

            orgs = result.get("organizations", []) or result.get("accounts", [])
            if not orgs:
                console.print("[yellow]No more results from Apollo.[/yellow]")
                break

            with engine.connect() as conn:
                for org in orgs:
                    name = org.get("name", "")
                    if not name:
                        continue

                    company_id = _make_id(name)
                    existing = conn.execute(
                        sa.select(companies.c.id).where(companies.c.id == company_id)
                    ).first()
                    if existing:
                        continue

                    website = org.get("website_url") or org.get("primary_domain") or ""
                    if website and not website.startswith("http"):
                        website = f"https://{website}"

                    linkedin_url = org.get("linkedin_url", "")
                    industry = org.get("industry", "")
                    size = org.get("estimated_num_employees")
                    funding = org.get("latest_funding_stage", "")

                    conn.execute(sa.insert(companies).values(
                        id=company_id,
                        name=name,
                        linkedin_url=linkedin_url,
                        website_url=website,
                        size_range=str(size) if size else None,
                        funding_stage=funding,
                        industry=industry,
                        scrape_status="done",
                        research_status="pending",
                        news_status="pending",
                    ))
                    total_companies += 1

                conn.commit()

            progress.update(task, completed=total_companies)
            console.print(f"  Page {page}: {len(orgs)} found (total: {total_companies})")

            # Check pagination
            pagination = result.get("pagination", {})
            total_available = pagination.get("total_entries", 0)
            if total_companies >= max_results or page * per_page >= total_available:
                break

            page += 1
            time.sleep(1)  # Rate limiting

    # ── Phase 2: Find decision makers ──
    console.print(f"\n[bold]Phase 2: Finding decision makers...[/bold]")

    with engine.connect() as conn:
        company_rows = conn.execute(
            sa.select(companies.c.id, companies.c.name).where(
                companies.c.scrape_status == "done"
            )
        ).fetchall()

    # Get ICP titles for filtering
    icp_titles = settings.get_titles()
    # Simplify titles for Apollo search (it does fuzzy matching)
    apollo_titles = list(set(
        t for t in ["CEO", "Founder", "CTO", "VP Engineering", "Head of Product", "Product Director"]
        if any(t.lower() in icp.lower() or icp.lower() in t.lower() for icp in icp_titles)
    )) or ["CEO", "Founder", "CTO", "VP Engineering", "Head of Product"]

    total_leads = 0

    with Progress() as progress:
        task = progress.add_task("Finding leads...", total=len(company_rows))

        # Process in batches to reduce API calls
        batch_size = 10
        for i in range(0, len(company_rows), batch_size):
            batch = company_rows[i:i + batch_size]
            batch_names = {row.name: row.id for row in batch}

            try:
                # Search people by title across these companies
                for company_row in batch:
                    result = _search_people(
                        titles=apollo_titles,
                        page=1,
                        per_page=10,
                    )

                    people = result.get("people", [])

                    with engine.connect() as conn:
                        for person in people:
                            person_name = person.get("name", "")
                            if not person_name:
                                continue

                            # Check this person belongs to one of our target companies
                            org = person.get("organization", {}) or {}
                            org_name = org.get("name", "")

                            # Match to our company
                            company_id = None
                            if org_name and org_name in batch_names:
                                company_id = batch_names[org_name]
                            elif org_name:
                                # Fuzzy match
                                for bn, bid in batch_names.items():
                                    if bn.lower() in org_name.lower() or org_name.lower() in bn.lower():
                                        company_id = bid
                                        break

                            if not company_id:
                                company_id = company_row.id

                            title = person.get("title", "")
                            persona = _map_persona(title)
                            if persona == "other":
                                continue

                            person_id = _make_id(f"{person_name}-{company_id}")
                            existing = conn.execute(
                                sa.select(persons.c.id).where(persons.c.id == person_id)
                            ).first()
                            if existing:
                                continue

                            email = person.get("email", "")
                            linkedin = person.get("linkedin_url", "")

                            conn.execute(sa.insert(persons).values(
                                id=person_id,
                                company_id=company_id,
                                name=person_name,
                                title=title,
                                linkedin_url=linkedin,
                                email=email,
                                persona=persona,
                            ))
                            total_leads += 1

                        conn.commit()

                    time.sleep(0.5)  # Rate limiting

            except httpx.HTTPStatusError as e:
                console.print(f"  [red]Apollo API error: {e.response.status_code}[/red]")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

            for _ in batch:
                progress.advance(task)

    console.print(f"\n[green]✓ Done![/green] {total_companies} companies, {total_leads} leads")
