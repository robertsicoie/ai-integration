"""Stage 3: Press and news research using DuckDuckGo search."""

import sqlalchemy as sa
from ddgs import DDGS
from rich.console import Console
from rich.progress import Progress

from prospect.config import settings
from prospect.db import companies, news_items, init_db

console = Console()

# Search queries designed to surface buying triggers
SEARCH_TEMPLATES = [
    '"{company}" funding OR raised OR investment OR Series',
    '"{company}" product launch OR announcement OR release',
    '"{company}" hiring OR engineering OR CTO OR team',
    '"{company}" platform OR modernization OR expansion',
]


def _search_news(company_name: str, max_results_per_query: int = 5) -> list[dict]:
    """Search for news about a company using DuckDuckGo."""
    all_results = []
    seen_urls = set()

    ddgs = DDGS()

    for template in SEARCH_TEMPLATES:
        query = template.format(company=company_name)

        # Try news search first
        try:
            results = list(ddgs.news(query, max_results=max_results_per_query))
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "headline": r.get("title", ""),
                        "url": url,
                        "date": r.get("date", ""),
                        "snippet": r.get("body", ""),
                    })
        except Exception:
            pass

        # Always also try text search for broader coverage
        try:
            results = list(ddgs.text(query, max_results=max_results_per_query))
            for r in results:
                url = r.get("href", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "headline": r.get("title", ""),
                        "url": url,
                        "date": "",
                        "snippet": r.get("body", ""),
                    })
        except Exception:
            pass

    return all_results


async def research_news(batch_size: int = 10):
    """Search for press articles and buying triggers for companies."""
    engine = init_db()

    console.print("[bold]Stage 3: News & Press Research[/bold]")

    with engine.connect() as conn:
        pending = conn.execute(
            sa.select(companies).where(
                companies.c.news_status == "pending",
                companies.c.research_status == "done",
            ).limit(batch_size)
        ).fetchall()

    if not pending:
        console.print("[yellow]No companies pending news research.[/yellow]")
        return

    console.print(f"Searching news for {len(pending)} companies...")

    with Progress() as progress:
        task = progress.add_task("Searching news...", total=len(pending))

        for row in pending:
            company_name = row.name
            company_id = row.id

            try:
                console.print(f"  Searching: {company_name}")
                results = _search_news(company_name)

                with engine.connect() as conn:
                    for r in results:
                        conn.execute(
                            sa.insert(news_items).values(
                                company_id=company_id,
                                headline=r["headline"],
                                url=r["url"],
                                date=r["date"],
                                snippet=r["snippet"],
                            )
                        )

                    conn.execute(
                        sa.update(companies)
                        .where(companies.c.id == company_id)
                        .values(news_status="done")
                    )
                    conn.commit()

                console.print(f"  [green]✓[/green] {company_name}: {len(results)} articles found")

            except Exception as e:
                console.print(f"  [red]✗[/red] {company_name}: {e}")
                with engine.connect() as conn:
                    conn.execute(
                        sa.update(companies)
                        .where(companies.c.id == company_id)
                        .values(news_status="error")
                    )
                    conn.commit()

            progress.advance(task)

    console.print(f"\n[green]✓ News research complete![/green]")
