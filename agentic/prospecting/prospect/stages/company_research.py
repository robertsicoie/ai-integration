"""Stage 2: Company website research using httpx + Claude summarization."""

import httpx
import sqlalchemy as sa
from anthropic import Anthropic
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress

from prospect.config import settings
from prospect.db import companies, init_db

console = Console()

COMPANY_SUMMARY_PROMPT = """\
You are a B2B sales research analyst. Analyze the following website content from {company_name} and extract structured intelligence.

Website content:
---
{content}
---

Provide your analysis as a structured response with these exact sections:

**Summary**: One clear paragraph describing what this company does, their main product/service, and their value proposition.

**Industry**: The primary industry vertical (e.g., fintech, logistics, SaaS, proptech, healthtech, gaming, manufacturing, marketplace, etc.)

**Target Customer**: Who their primary customers are (e.g., "SMB retailers", "enterprise healthcare providers", etc.)

**Products/Services**: Bullet list of their key products or services.

**Tech Stack**: Any technology, platform, or technical capabilities mentioned (e.g., "AI-powered", "cloud-native", "blockchain", etc.). If none mentioned, say "Not specified".
"""

PAGES_TO_SCRAPE = ["/", "/about", "/about-us", "/product", "/products", "/platform", "/pricing"]


async def _fetch_website(url: str) -> str:
    """Fetch and extract visible text from a website."""
    all_text = []

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    ) as client:
        for path in PAGES_TO_SCRAPE:
            try:
                full_url = url.rstrip("/") + path
                resp = await client.get(full_url)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "lxml")

                # Remove non-content elements
                for tag in soup.find_all(["nav", "footer", "script", "style", "header"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)
                # Limit per page
                if text and len(text) > 100:
                    all_text.append(f"--- Page: {path} ---\n{text[:3000]}")

            except Exception:
                continue

    return "\n\n".join(all_text)[:10000]  # Cap total content


def _summarize_with_claude(company_name: str, content: str) -> dict:
    """Use Claude to summarize company website content."""
    client = Anthropic(api_key=settings.anthropic_api_key)

    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": COMPANY_SUMMARY_PROMPT.format(
                    company_name=company_name, content=content
                ),
            }
        ],
    )

    response_text = message.content[0].text

    # Parse structured sections from response
    result = {
        "summary": "",
        "industry_detected": "",
        "target_customer": "",
        "products_services": "",
        "tech_stack": "",
    }

    sections = {
        "**Summary**:": "summary",
        "**Industry**:": "industry_detected",
        "**Target Customer**:": "target_customer",
        "**Products/Services**:": "products_services",
        "**Tech Stack**:": "tech_stack",
    }

    for marker, field in sections.items():
        if marker in response_text:
            start = response_text.index(marker) + len(marker)
            # Find next section or end
            end = len(response_text)
            for other_marker in sections:
                if other_marker != marker and other_marker in response_text:
                    other_start = response_text.index(other_marker)
                    if other_start > start and other_start < end:
                        end = other_start
            result[field] = response_text[start:end].strip()

    return result


async def research_companies(batch_size: int = 10):
    """Research company websites and summarize with Claude."""
    engine = init_db()

    console.print("[bold]Stage 2: Company Website Research[/bold]")

    with engine.connect() as conn:
        pending = conn.execute(
            sa.select(companies).where(companies.c.research_status == "pending").limit(batch_size)
        ).fetchall()

    if not pending:
        console.print("[yellow]No companies pending research.[/yellow]")
        return

    console.print(f"Researching {len(pending)} companies...")

    with Progress() as progress:
        task = progress.add_task("Researching...", total=len(pending))

        for row in pending:
            company_name = row.name
            website_url = row.website_url

            if not website_url:
                # Try to construct URL from company name
                slug = company_name.lower().replace(" ", "").replace(",", "").replace(".", "")
                website_url = f"https://www.{slug}.com"

            try:
                console.print(f"  Fetching {company_name} ({website_url})...")
                content = await _fetch_website(website_url)

                if not content or len(content) < 100:
                    console.print(f"  [yellow]No content found for {company_name}[/yellow]")
                    with engine.connect() as conn:
                        conn.execute(
                            sa.update(companies)
                            .where(companies.c.id == row.id)
                            .values(research_status="error", summary="No website content found")
                        )
                        conn.commit()
                    progress.advance(task)
                    continue

                console.print(f"  Summarizing with Claude...")
                result = _summarize_with_claude(company_name, content)

                with engine.connect() as conn:
                    conn.execute(
                        sa.update(companies)
                        .where(companies.c.id == row.id)
                        .values(
                            research_status="done",
                            website_url=website_url,
                            **result,
                        )
                    )
                    conn.commit()

                console.print(f"  [green]✓[/green] {company_name}: {result['industry_detected']}")

            except Exception as e:
                console.print(f"  [red]✗[/red] {company_name}: {e}")
                with engine.connect() as conn:
                    conn.execute(
                        sa.update(companies)
                        .where(companies.c.id == row.id)
                        .values(research_status="error", summary=str(e))
                    )
                    conn.commit()

            progress.advance(task)

    console.print(f"\n[green]✓ Research complete![/green]")
