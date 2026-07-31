"""CLI entry point for the prospecting pipeline."""

import asyncio
import csv
import json
import sys

import sqlalchemy as sa
import typer
from rich.console import Console
from rich.table import Table

from prospect.db import companies, persons, news_items, draft_emails, init_db

app = typer.Typer(
    name="prospect",
    help="B2B prospecting pipeline: LinkedIn -> Research -> News -> Email",
)
console = Console()


@app.command()
def scrape(
    max_results: int = typer.Option(200, help="Max companies to scrape"),
    headless: bool = typer.Option(False, help="Run browser in headless mode"),
    source: str = typer.Option("linkedin", help="Source: linkedin or apollo"),
):
    """Stage 1: Scrape for leads matching ICP (LinkedIn or Apollo.io)."""
    if source == "apollo":
        from prospect.stages.apollo import scrape_apollo
        asyncio.run(scrape_apollo(max_results=max_results))
    else:
        from prospect.stages.linkedin import scrape_linkedin
        asyncio.run(scrape_linkedin(max_results=max_results, headless=headless))


@app.command()
def research(
    batch_size: int = typer.Option(10, help="Number of companies per batch"),
):
    """Stage 2: Research company websites and summarize with Claude."""
    from prospect.stages.company_research import research_companies

    asyncio.run(research_companies(batch_size=batch_size))


@app.command()
def news(
    batch_size: int = typer.Option(10, help="Number of companies per batch"),
):
    """Stage 3: Search press articles for funding, hiring, and buying triggers."""
    from prospect.stages.news_research import research_news

    asyncio.run(research_news(batch_size=batch_size))


@app.command()
def draft(
    batch_size: int = typer.Option(10, help="Number of emails per batch"),
    regenerate: bool = typer.Option(False, help="Regenerate existing drafts"),
):
    """Stage 4: Draft personalized outreach emails with Claude."""
    from prospect.stages.email_draft import draft_outreach_emails

    asyncio.run(draft_outreach_emails(batch_size=batch_size, regenerate=regenerate))


@app.command()
def qualify(
    linkedin_url: str = typer.Argument(help="LinkedIn profile URL (public or Sales Navigator)"),
    headless: bool = typer.Option(False, help="Run browser in headless mode"),
    skip_message: bool = typer.Option(False, "--skip-message", help="Only check ICP fit, don't draft a message"),
):
    """Check if a LinkedIn profile qualifies as a lead and draft a connection message."""
    from prospect.stages.linkedin_qualify import qualify_profile

    asyncio.run(qualify_profile(
        linkedin_url=linkedin_url,
        headless=headless,
        skip_message=skip_message,
    ))


STAGE_ORDER = ["scrape", "research", "news", "draft"]


@app.command()
def run(
    from_stage: str = typer.Option("scrape", "--from", help="Stage to start from: scrape, research, news, or draft"),
    max_results: int = typer.Option(200, help="Max companies to scrape (stage 1)"),
    batch_size: int = typer.Option(10, help="Batch size for research/news/draft stages"),
    source: str = typer.Option("linkedin", help="Scrape source: linkedin or apollo (stage 1)"),
    headless: bool = typer.Option(False, help="Run browser in headless mode (stage 1)"),
    regenerate: bool = typer.Option(False, help="Regenerate existing email drafts (stage 4)"),
):
    """Run the pipeline from a given stage through to the end.

    Examples:
      prospect run                              # full pipeline from scrape
      prospect run --from research              # resume from company research
      prospect run --from news                  # resume from news search
      prospect run --from scrape --source apollo  # use Apollo.io instead of LinkedIn
    """
    from prospect.stages.company_research import research_companies
    from prospect.stages.email_draft import draft_outreach_emails
    from prospect.stages.news_research import research_news

    if from_stage not in STAGE_ORDER:
        console.print(f"[red]Invalid stage: {from_stage}. Choose from: {', '.join(STAGE_ORDER)}[/red]")
        sys.exit(1)

    start_idx = STAGE_ORDER.index(from_stage)
    stages_to_run = STAGE_ORDER[start_idx:]

    async def _run():
        console.print(f"[bold]Running pipeline: {' -> '.join(stages_to_run)}[/bold]\n")

        for stage_name in stages_to_run:
            stage_num = STAGE_ORDER.index(stage_name) + 1
            total = len(STAGE_ORDER)

            if stage_name == "scrape":
                if source == "apollo":
                    from prospect.stages.apollo import scrape_apollo
                    console.print(f"[bold cyan]{stage_num}/{total}[/bold cyan] Apollo.io Prospecting")
                    await scrape_apollo(max_results=max_results)
                else:
                    from prospect.stages.linkedin import scrape_linkedin
                    console.print(f"[bold cyan]{stage_num}/{total}[/bold cyan] LinkedIn Scraping")
                    await scrape_linkedin(max_results=max_results, headless=headless)

            elif stage_name == "research":
                console.print(f"[bold cyan]{stage_num}/{total}[/bold cyan] Company Research")
                await research_companies(batch_size=batch_size)

            elif stage_name == "news":
                console.print(f"[bold cyan]{stage_num}/{total}[/bold cyan] News Research")
                await research_news(batch_size=batch_size)

            elif stage_name == "draft":
                console.print(f"[bold cyan]{stage_num}/{total}[/bold cyan] Email Drafting")
                await draft_outreach_emails(batch_size=batch_size, regenerate=regenerate)

            console.print()

        console.print("[bold green]Pipeline complete![/bold green]")

    asyncio.run(_run())


@app.command()
def run_all(
    max_results: int = typer.Option(200, help="Max companies to scrape"),
    batch_size: int = typer.Option(10, help="Batch size for research stages"),
):
    """Run the full pipeline end-to-end (alias for 'run --from scrape')."""
    from prospect.stages.company_research import research_companies
    from prospect.stages.email_draft import draft_outreach_emails
    from prospect.stages.linkedin import scrape_linkedin
    from prospect.stages.news_research import research_news

    async def _run():
        console.print("[bold]Running full prospecting pipeline[/bold]\n")

        console.print("[bold cyan]1/4[/bold cyan] LinkedIn Scraping")
        await scrape_linkedin(max_results=max_results)

        console.print(f"\n[bold cyan]2/4[/bold cyan] Company Research")
        await research_companies(batch_size=batch_size)

        console.print(f"\n[bold cyan]3/4[/bold cyan] News Research")
        await research_news(batch_size=batch_size)

        console.print(f"\n[bold cyan]4/4[/bold cyan] Email Drafting")
        await draft_outreach_emails(batch_size=batch_size)

        console.print("\n[bold green]Pipeline complete![/bold green]")

    asyncio.run(_run())


@app.command()
def status():
    """Show pipeline progress — counts by status for each stage."""
    engine = init_db()

    with engine.connect() as conn:
        # Company counts by status
        table = Table(title="Pipeline Status")
        table.add_column("Stage", style="bold")
        table.add_column("Pending", style="yellow")
        table.add_column("Done", style="green")
        table.add_column("Error", style="red")
        table.add_column("Total")

        for stage_col, label in [
            ("scrape_status", "1. LinkedIn Scrape"),
            ("research_status", "2. Company Research"),
            ("news_status", "3. News Research"),
        ]:
            col = getattr(companies.c, stage_col)
            counts = {}
            for status_val in ["pending", "done", "error"]:
                count = conn.execute(
                    sa.select(sa.func.count()).where(col == status_val)
                ).scalar()
                counts[status_val] = count

            total = sum(counts.values())
            table.add_row(
                label,
                str(counts.get("pending", 0)),
                str(counts.get("done", 0)),
                str(counts.get("error", 0)),
                str(total),
            )

        # Email drafts
        draft_count = conn.execute(sa.select(sa.func.count()).select_from(draft_emails)).scalar()
        person_count = conn.execute(sa.select(sa.func.count()).select_from(persons)).scalar()
        table.add_row(
            "4. Email Drafts",
            str(person_count - draft_count),
            str(draft_count),
            "—",
            str(person_count),
        )

        # Total persons
        total_persons = conn.execute(sa.select(sa.func.count()).select_from(persons)).scalar()
        total_news = conn.execute(sa.select(sa.func.count()).select_from(news_items)).scalar()

    console.print(table)
    console.print(f"\nLeads: {total_persons} | News articles: {total_news}")


def _split_name(full_name: str) -> tuple[str, str]:
    """Split a full name into (first_name, last_name)."""
    parts = (full_name or "").strip().split()
    if len(parts) == 0:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], " ".join(parts[1:]))


def _build_news_note(conn, company_id: str) -> str:
    """Build a combined note from news items for a company."""
    rows = conn.execute(
        sa.select(news_items.c.headline, news_items.c.url, news_items.c.date, news_items.c.snippet)
        .where(news_items.c.company_id == company_id)
    ).fetchall()
    if not rows:
        return ""
    lines = []
    for r in rows:
        date_str = f" ({r.date})" if r.date else ""
        lines.append(f"- {r.headline}{date_str}: {r.snippet[:200]}")
        if r.url:
            lines.append(f"  {r.url}")
    return "\n".join(lines)


@app.command()
def export(
    format: str = typer.Option("csv", help="Export format: csv, json, hubspot, instantly, or lemlist"),
    output: str = typer.Option("prospect_export", help="Output filename (without extension)"),
):
    """Export all pipeline data: companies, leads, news, and emails.

    Formats:
      csv       — Separate CSV files for companies, leads, news, emails
      json      — Single JSON file with all data
      hubspot   — HubSpot-compatible CSVs (companies + contacts)
      instantly — Instantly.io campaign CSV (contacts + personalization)
      lemlist   — Lemlist campaign CSV (contacts + personalization)
    """
    engine = init_db()

    with engine.connect() as conn:
        # ── Companies ──
        company_rows = conn.execute(
            sa.select(companies).order_by(companies.c.name)
        ).fetchall()
        company_cols = [
            "id", "name", "linkedin_url", "website_url", "size_range",
            "funding_stage", "industry", "summary", "industry_detected",
            "target_customer", "products_services", "tech_stack",
            "scrape_status", "research_status", "news_status",
        ]

        # ── Leads (persons + company info) ──
        lead_rows = conn.execute(
            sa.select(
                persons,
                companies.c.name.label("company_name"),
                companies.c.website_url.label("company_website"),
                companies.c.industry_detected.label("company_industry"),
                companies.c.summary.label("company_summary"),
            )
            .outerjoin(companies, persons.c.company_id == companies.c.id)
            .order_by(companies.c.name, persons.c.name)
        ).fetchall()
        lead_cols = [
            "id", "company_id", "company_name", "name", "title",
            "linkedin_url", "email", "persona",
        ]

        # ── News ──
        news_rows = conn.execute(
            sa.select(
                news_items,
                companies.c.name.label("company_name"),
            )
            .outerjoin(companies, news_items.c.company_id == companies.c.id)
            .order_by(companies.c.name, news_items.c.date.desc())
        ).fetchall()
        news_cols = [
            "company_id", "company_name", "headline", "url", "date",
            "snippet", "full_summary",
        ]

        # ── Emails ──
        email_rows = conn.execute(
            sa.select(
                draft_emails.c.person_id,
                draft_emails.c.subject,
                draft_emails.c.body,
                draft_emails.c.persona_template,
                draft_emails.c.version,
                persons.c.name.label("person_name"),
                persons.c.title.label("person_title"),
                persons.c.linkedin_url.label("person_linkedin"),
                persons.c.email.label("person_email"),
                persons.c.company_id,
                companies.c.name.label("company_name"),
                companies.c.website_url,
                companies.c.industry_detected,
            )
            .join(persons, draft_emails.c.person_id == persons.c.id)
            .join(companies, draft_emails.c.company_id == companies.c.id)
            .order_by(companies.c.name, persons.c.name, draft_emails.c.version.desc())
        ).fetchall()
        email_cols = [
            "company_name", "person_name", "person_title", "person_linkedin",
            "person_email", "industry", "subject", "body", "persona",
            "version", "website",
        ]

    total = len(company_rows) + len(lead_rows) + len(news_rows) + len(email_rows)
    if total == 0:
        console.print("[yellow]No data to export.[/yellow]")
        return

    def _row_to_dict(row, cols):
        return {col: getattr(row, col, None) for col in cols}

    # ── HubSpot format ──
    if format == "hubspot":
        exported = []

        # 1. Companies CSV — maps to HubSpot Company properties
        if company_rows:
            fname = f"{output}_hubspot_companies.csv"
            hs_company_cols = [
                "Company Name", "Company Domain Name", "LinkedIn Company Page",
                "Industry", "Number of Employees", "Description",
                "Funding Stage", "Target Customer", "Products/Services",
                "Technology Stack",
            ]
            with open(fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=hs_company_cols)
                writer.writeheader()
                for row in company_rows:
                    # Extract domain from website URL
                    website = row.website_url or ""
                    domain = website.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")

                    writer.writerow({
                        "Company Name": row.name,
                        "Company Domain Name": domain,
                        "LinkedIn Company Page": row.linkedin_url or "",
                        "Industry": row.industry_detected or row.industry or "",
                        "Number of Employees": row.size_range or "",
                        "Description": row.summary or "",
                        "Funding Stage": row.funding_stage or "",
                        "Target Customer": row.target_customer or "",
                        "Products/Services": row.products_services or "",
                        "Technology Stack": row.tech_stack or "",
                    })
            exported.append(f"  {fname} ({len(company_rows)} companies)")

        # 2. Contacts CSV — maps to HubSpot Contact properties
        #    Includes associated company name for HubSpot association on import
        if lead_rows:
            fname = f"{output}_hubspot_contacts.csv"

            # Build a lookup: person_id -> latest email draft
            email_by_person = {}
            for row in email_rows:
                pid = row.person_id
                if pid not in email_by_person:  # first = latest version (ordered desc)
                    email_by_person[pid] = row

            hs_contact_cols = [
                "First Name", "Last Name", "Email", "Job Title",
                "LinkedIn URL", "Company Name", "Lifecycle Stage", "Lead Status",
                "Buyer Persona", "Company Summary", "News & Triggers",
                "Draft Email Subject", "Draft Email Body",
            ]
            with open(fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=hs_contact_cols)
                writer.writeheader()
                for row in lead_rows:
                    first, last = _split_name(row.name)
                    draft = email_by_person.get(row.id)

                    # Build news note
                    with engine.connect() as c2:
                        news_note = _build_news_note(c2, row.company_id)

                    writer.writerow({
                        "First Name": first,
                        "Last Name": last,
                        "Email": row.email or "",
                        "Job Title": row.title or "",
                        "LinkedIn URL": row.linkedin_url or "",
                        "Company Name": row.company_name or "",
                        "Lifecycle Stage": "lead",
                        "Lead Status": "New",
                        "Buyer Persona": row.persona or "",
                        "Company Summary": row.company_summary or "",
                        "News & Triggers": news_note,
                        "Draft Email Subject": draft.subject if draft else "",
                        "Draft Email Body": draft.body if draft else "",
                    })
            exported.append(f"  {fname} ({len(lead_rows)} contacts)")

        if exported:
            console.print("[green]Exported for HubSpot:[/green]")
            for line in exported:
                console.print(line)
            console.print()
            console.print("[bold]HubSpot import instructions:[/bold]")
            console.print("  1. Go to HubSpot > Contacts > Import")
            console.print("  2. Choose 'File from computer' > 'Multiple files with associations'")
            console.print("  3. Upload companies file first, then contacts file")
            console.print("  4. Map 'Company Name' as the association key between both files")
            console.print("  5. Map columns to HubSpot properties (most will auto-match)")
            console.print("  6. 'News & Triggers', 'Draft Email Subject/Body' map to custom properties")
            console.print("     or create them as single-line/multi-line text in HubSpot first")
        else:
            console.print("[yellow]No data to export.[/yellow]")
        return

    # ── Instantly.io format ──
    if format == "instantly":
        if not lead_rows:
            console.print("[yellow]No contacts to export.[/yellow]")
            return

        fname = f"{output}_instantly.csv"
        # Instantly CSV: email, first_name, last_name, company_name, + custom columns
        hs_cols = [
            "email", "first_name", "last_name", "company_name",
            "personalization", "phone", "website",
            "custom1", "custom2", "custom3", "custom4",
        ]

        # Build email draft lookup
        email_by_person = {}
        for row in email_rows:
            pid = row.person_id
            if pid not in email_by_person:
                email_by_person[pid] = row

        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=hs_cols)
            writer.writeheader()
            for row in lead_rows:
                first, last = _split_name(row.name)
                draft = email_by_person.get(row.id)

                # Build personalization snippet (first line of email body)
                personalization = ""
                if draft and draft.body:
                    # First sentence of the email
                    lines = [l.strip() for l in draft.body.split("\n") if l.strip()]
                    personalization = lines[0] if lines else ""

                writer.writerow({
                    "email": row.email or "",
                    "first_name": first,
                    "last_name": last,
                    "company_name": row.company_name or "",
                    "personalization": personalization,
                    "phone": "",
                    "website": getattr(row, "company_website", "") or "",
                    "custom1": row.title or "",              # Job Title
                    "custom2": row.linkedin_url or "",       # LinkedIn URL
                    "custom3": draft.subject if draft else "",  # Draft Subject
                    "custom4": draft.body if draft else "",   # Full Draft Body
                })

        console.print(f"[green]Exported for Instantly.io:[/green]")
        console.print(f"  {fname} ({len(lead_rows)} contacts)")
        console.print()
        console.print("[bold]Instantly import instructions:[/bold]")
        console.print("  1. Go to Instantly > Campaigns > select/create campaign")
        console.print("  2. Click 'Upload Leads' > choose the CSV file")
        console.print("  3. Map columns: email, first_name, last_name, company_name auto-match")
        console.print("  4. Map custom1=Job Title, custom2=LinkedIn, custom3=Subject, custom4=Body")
        console.print("  5. Use {{personalization}} in your email template for the opening line")
        console.print("  6. Use {{custom1}}, {{custom2}}, etc. for other variables")
        return

    # ── Lemlist format ──
    if format == "lemlist":
        if not lead_rows:
            console.print("[yellow]No contacts to export.[/yellow]")
            return

        fname = f"{output}_lemlist.csv"
        # Lemlist CSV: firstName, lastName, email, companyName, + custom columns
        lem_cols = [
            "firstName", "lastName", "email", "companyName",
            "linkedinUrl", "jobTitle", "icebreaker",
            "companyWebsite", "companyIndustry", "companyDescription",
            "emailSubject", "emailBody",
        ]

        email_by_person = {}
        for row in email_rows:
            pid = row.person_id
            if pid not in email_by_person:
                email_by_person[pid] = row

        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=lem_cols)
            writer.writeheader()
            for row in lead_rows:
                first, last = _split_name(row.name)
                draft = email_by_person.get(row.id)

                # Icebreaker: first paragraph of draft email
                icebreaker = ""
                if draft and draft.body:
                    paragraphs = [p.strip() for p in draft.body.split("\n\n") if p.strip()]
                    icebreaker = paragraphs[0] if paragraphs else ""

                writer.writerow({
                    "firstName": first,
                    "lastName": last,
                    "email": row.email or "",
                    "companyName": row.company_name or "",
                    "linkedinUrl": row.linkedin_url or "",
                    "jobTitle": row.title or "",
                    "icebreaker": icebreaker,
                    "companyWebsite": getattr(row, "company_website", "") or "",
                    "companyIndustry": getattr(row, "company_industry", "") or "",
                    "companyDescription": getattr(row, "company_summary", "") or "",
                    "emailSubject": draft.subject if draft else "",
                    "emailBody": draft.body if draft else "",
                })

        console.print(f"[green]Exported for Lemlist:[/green]")
        console.print(f"  {fname} ({len(lead_rows)} contacts)")
        console.print()
        console.print("[bold]Lemlist import instructions:[/bold]")
        console.print("  1. Go to Lemlist > Campaigns > select/create campaign")
        console.print("  2. Click 'Import leads' > 'From CSV' > upload the file")
        console.print("  3. Columns auto-map: firstName, lastName, email, companyName")
        console.print("  4. Use {{icebreaker}} in your template for the personalized opening")
        console.print("  5. Use {{jobTitle}}, {{linkedinUrl}}, etc. for other variables")
        console.print("  6. emailSubject/emailBody are pre-drafted — use as reference or template")
        return

    # ── CSV format ──
    if format == "csv":
        exported = []

        if company_rows:
            fname = f"{output}_companies.csv"
            with open(fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=company_cols)
                writer.writeheader()
                for row in company_rows:
                    writer.writerow(_row_to_dict(row, company_cols))
            exported.append(f"  {fname} ({len(company_rows)} companies)")

        if lead_rows:
            fname = f"{output}_leads.csv"
            with open(fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=lead_cols)
                writer.writeheader()
                for row in lead_rows:
                    writer.writerow(_row_to_dict(row, lead_cols))
            exported.append(f"  {fname} ({len(lead_rows)} leads)")

        if news_rows:
            fname = f"{output}_news.csv"
            with open(fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=news_cols)
                writer.writeheader()
                for row in news_rows:
                    writer.writerow(_row_to_dict(row, news_cols))
            exported.append(f"  {fname} ({len(news_rows)} articles)")

        if email_rows:
            fname = f"{output}_emails.csv"
            email_col_map = {
                "company_name": "company_name", "person_name": "person_name",
                "person_title": "person_title", "person_linkedin": "person_linkedin",
                "person_email": "person_email", "industry": "industry_detected",
                "subject": "subject", "body": "body", "persona": "persona_template",
                "version": "version", "website": "website_url",
            }
            with open(fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=email_cols)
                writer.writeheader()
                for row in email_rows:
                    writer.writerow({col: getattr(row, attr, None) for col, attr in email_col_map.items()})
            exported.append(f"  {fname} ({len(email_rows)} emails)")

        console.print(f"[green]Exported:[/green]")
        for line in exported:
            console.print(line)

    elif format == "json":
        filename = f"{output}.json"
        data = {
            "companies": [_row_to_dict(r, company_cols) for r in company_rows],
            "leads": [_row_to_dict(r, lead_cols) for r in lead_rows],
            "news": [_row_to_dict(r, news_cols) for r in news_rows],
            "emails": [
                {
                    "company_name": r.company_name, "person_name": r.person_name,
                    "person_title": r.person_title, "person_linkedin": r.person_linkedin,
                    "person_email": r.person_email, "industry": r.industry_detected,
                    "subject": r.subject, "body": r.body, "persona": r.persona_template,
                    "version": r.version, "website": r.website_url,
                }
                for r in email_rows
            ],
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(
            f"[green]Exported to {filename}:[/green] "
            f"{len(company_rows)} companies, {len(lead_rows)} leads, "
            f"{len(news_rows)} articles, {len(email_rows)} emails"
        )

    else:
        console.print(f"[red]Unknown format: {format}. Use csv, json, hubspot, instantly, or lemlist.[/red]")
        sys.exit(1)


@app.command()
def reset(
    stage: str = typer.Argument(help="Stage to reset: scrape, research, news, draft, or all"),
):
    """Reset a stage so it can be re-run from scratch."""
    engine = init_db()
    valid_stages = ["scrape", "research", "news", "draft", "all"]
    if stage not in valid_stages:
        console.print(f"[red]Invalid stage: {stage}. Choose from: {', '.join(valid_stages)}[/red]")
        sys.exit(1)

    with engine.connect() as conn:
        if stage in ("research", "all"):
            conn.execute(sa.update(companies).values(
                research_status="pending", summary=None, industry_detected=None,
                target_customer=None, products_services=None, tech_stack=None,
            ))
            console.print("  Reset: company research")

        if stage in ("news", "all"):
            conn.execute(sa.update(companies).values(news_status="pending"))
            conn.execute(sa.delete(news_items))
            console.print("  Reset: news research")

        if stage in ("draft", "all"):
            conn.execute(sa.delete(draft_emails))
            console.print("  Reset: email drafts")

        if stage in ("scrape", "all"):
            conn.execute(sa.delete(draft_emails))
            conn.execute(sa.delete(news_items))
            conn.execute(sa.delete(persons))
            conn.execute(sa.delete(companies))
            console.print("  Reset: all scraped data (companies, leads, news, emails)")

        conn.commit()

    console.print(f"[green]Stage '{stage}' reset successfully.[/green]")


@app.command()
def init():
    """Initialize the database (creates tables if needed)."""
    init_db()
    console.print("[green]Database initialized.[/green]")


@app.command(name="help")
def show_help():
    """Show pipeline overview and available commands."""
    console.print()
    console.print("[bold]Prospect — B2B Prospecting Pipeline[/bold]")
    console.print("LinkedIn -> Company Research -> News -> Email Drafting\n")

    table = Table(show_header=True, header_style="bold", show_lines=False, pad_edge=False)
    table.add_column("Command", style="cyan", min_width=12)
    table.add_column("Stage", min_width=6)
    table.add_column("Description")
    table.add_column("Key Options", style="dim")

    table.add_row("init", "—", "Initialize the database", "")
    table.add_row("scrape", "1", "Scrape for companies & leads (LinkedIn or Apollo)", "--max-results, --source")
    table.add_row("research", "2", "Research company websites and summarize with Claude", "--batch-size")
    table.add_row("news", "3", "Search press articles for funding, hiring, triggers", "--batch-size")
    table.add_row("draft", "4", "Draft personalized outreach emails with Claude", "--batch-size, --regenerate")
    table.add_row("run", "1-4", "Run pipeline from a stage to the end", "--from, --batch-size, --max-results")
    table.add_row("run-all", "1-4", "Run full pipeline (alias for run)", "--max-results, --batch-size")
    table.add_row("qualify", "—", "Check LinkedIn profile ICP fit + draft message", "<url>, --skip-message")
    table.add_row("status", "—", "Show pipeline progress by stage", "")
    table.add_row("export", "—", "Export data (csv/json/hubspot/instantly/lemlist)", "--format, --output")
    table.add_row("reset", "—", "Reset a stage to re-run it", "scrape|research|news|draft|all")
    table.add_row("help", "—", "Show this help message", "")

    console.print(table)

    console.print("\n[bold]Typical workflow:[/bold]")
    console.print("  [cyan]prospect run --from scrape --batch-size 20[/cyan]  # full pipeline")
    console.print("  [cyan]prospect run --from research[/cyan]               # resume from research onward")
    console.print("  [cyan]prospect run --from news[/cyan]                   # resume from news onward")
    console.print("  [cyan]prospect export --format csv[/cyan]               # export results")
    console.print()


if __name__ == "__main__":
    app()
