"""LinkedIn profile qualification: scrape profile, check ICP fit, draft message."""

import asyncio
import json
import re

import httpx
import sqlalchemy as sa
from anthropic import Anthropic
from bs4 import BeautifulSoup
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prospect.browser import create_browser, new_stealth_page, random_delay, wait_for_linkedin_login
from prospect.config import settings
from prospect.db import companies, persons, news_items, init_db
from prospect.stages.email_draft import PERSONA_CONTEXT
from prospect.stages.linkedin import _map_persona

console = Console()


# ── Profile scraping ──────────────────────────────────────────────────────────

async def scrape_linkedin_profile(page, url: str) -> dict:
    """Navigate to a LinkedIn profile and extract key data.

    Works with both public LinkedIn profiles and Sales Navigator URLs.
    Returns dict with: name, title, company, location, linkedin_url, about.
    """
    is_sales_nav = "/sales/" in url

    await page.goto(url)
    await asyncio.sleep(4)

    profile = {
        "name": None,
        "title": None,
        "company": None,
        "location": None,
        "linkedin_url": url,
        "about": None,
    }

    if is_sales_nav:
        profile = await _scrape_sales_nav_profile(page, profile)
    else:
        profile = await _scrape_public_profile(page, profile)

    # Classify persona from title
    if profile["title"]:
        profile["persona"] = _map_persona(profile["title"])
    else:
        profile["persona"] = "other"

    return profile


async def _scrape_sales_nav_profile(page, profile: dict) -> dict:
    """Extract profile data from a Sales Navigator profile page."""
    # Name
    for sel in [
        "span[data-anonymize='person-name']",
        "h1",
        "[class*='profile-topcard'] [class*='name']",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text and len(text) < 100:
                profile["name"] = text
                break

    # Title / headline
    for sel in [
        "span[data-anonymize='title']",
        "[class*='profile-topcard'] [class*='headline']",
        "[class*='profile-topcard__summary-position']",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text:
                profile["title"] = text
                break

    # Current company
    for sel in [
        "span[data-anonymize='company-name']",
        "a[href*='/sales/company/']",
        "[class*='profile-topcard__summary-position'] + *",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text and len(text) < 200:
                profile["company"] = text
                break

    # Location
    for sel in [
        "[class*='profile-topcard__location']",
        "span[data-anonymize='location']",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text:
                profile["location"] = text
                break

    # About / summary section
    for sel in [
        "[class*='profile-summary']",
        "[class*='about-section']",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text and len(text) > 20:
                profile["about"] = text[:1000]
                break

    return profile


async def _scrape_public_profile(page, profile: dict) -> dict:
    """Extract profile data from a public LinkedIn profile page."""
    # Name
    for sel in [
        "h1.text-heading-xlarge",
        "h1[class*='top-card-layout__title']",
        "h1",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text and len(text) < 100:
                profile["name"] = text
                break

    # Title / headline
    for sel in [
        "div.text-body-medium",
        "[class*='top-card-layout__headline']",
        "[class*='headline']",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text:
                profile["title"] = text
                break

    # Current company — parse from experience section or headline
    for sel in [
        "[class*='experience-item'] [class*='subtitle']",
        "button[aria-label*='Current company']",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text and len(text) < 200:
                profile["company"] = text
                break

    # If no company found, try to extract from title/headline
    if not profile["company"] and profile["title"]:
        # Titles often include "CTO at CompanyName"
        at_match = re.search(r"\bat\b\s+(.+)", profile["title"], re.IGNORECASE)
        if at_match:
            profile["company"] = at_match.group(1).strip()

    # Location
    for sel in [
        "span.text-body-small[class*='top-card']",
        "[class*='top-card-layout__first-subline'] span",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text:
                profile["location"] = text
                break

    # About
    for sel in [
        "#about + div",
        "section.pv-about-section",
        "[class*='summary']",
    ]:
        el = await page.query_selector(sel)
        if el:
            text = (await el.inner_text()).strip()
            if text and len(text) > 20:
                profile["about"] = text[:1000]
                break

    return profile


# ── ICP evaluation ────────────────────────────────────────────────────────────

ICP_EVALUATION_PROMPT = """\
You are a B2B sales qualification expert. Evaluate whether this LinkedIn profile matches the Ideal Customer Profile (ICP).

## Profile
- **Name**: {name}
- **Title**: {title}
- **Company**: {company}
- **Location**: {location}
- **About**: {about}

## ICP Criteria
- **Target titles**: {icp_titles}
- **Target industries**: {icp_industries}
- **Company sizes**: {icp_sizes}
- **Funding stages**: {icp_funding}
- **Locations**: {icp_locations}
- **Excluded industries**: {icp_exclude}

## Instructions
Evaluate how well this person matches the ICP. Consider:
1. Does their title match or closely relate to the target buyer personas?
2. Based on their company and headline, does their company likely operate in a target industry?
3. Any signals about company size or stage from their profile?
4. Location fit (if location criteria are specified)?

Return your response as valid JSON with this exact structure:
{{"qualified": true/false, "score": 0-100, "reasons": ["reason 1", "reason 2"], "gaps": ["gap 1"], "company_industry_guess": "best guess of their company's industry"}}

Be generous with qualification — if title matches and industry seems plausible, qualify them. Only disqualify if there are clear mismatches (e.g. wrong role type entirely, clearly excluded industry).
"""


def check_icp_fit(profile: dict) -> dict:
    """Use Claude to evaluate whether a profile matches the ICP.

    Returns dict with: qualified, score, reasons, gaps, company_industry_guess.
    """
    client = Anthropic(api_key=settings.anthropic_api_key)

    prompt = ICP_EVALUATION_PROMPT.format(
        name=profile.get("name") or "Unknown",
        title=profile.get("title") or "Unknown",
        company=profile.get("company") or "Unknown",
        location=profile.get("location") or "Unknown",
        about=profile.get("about") or "No about section available",
        icp_titles=", ".join(settings.get_titles()),
        icp_industries=", ".join(settings.get_industries()),
        icp_sizes=", ".join(settings.get_company_sizes()),
        icp_funding=", ".join(settings.get_funding_stages()),
        icp_locations=", ".join(settings.get_locations()) or "worldwide (no restriction)",
        icp_exclude=", ".join(settings.get_exclude_industries()) or "none",
    )

    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text

    # Parse JSON from response (handle markdown code fences)
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "qualified": False,
        "score": 0,
        "reasons": ["Could not parse evaluation"],
        "gaps": ["Evaluation failed"],
        "company_industry_guess": "unknown",
    }


# ── Company intelligence gathering ───────────────────────────────────────────

def _lookup_company_in_db(company_name: str) -> dict | None:
    """Check if we already have research on this company in the DB."""
    if not company_name:
        return None

    engine = init_db()
    with engine.connect() as conn:
        # Fuzzy match on company name
        row = conn.execute(
            sa.select(companies).where(
                sa.func.lower(companies.c.name) == company_name.lower()
            )
        ).first()

        if not row:
            # Try partial match
            row = conn.execute(
                sa.select(companies).where(
                    sa.func.lower(companies.c.name).contains(company_name.lower())
                )
            ).first()

        if row and row.research_status == "done":
            # Also grab news
            news_rows = conn.execute(
                sa.select(news_items).where(news_items.c.company_id == row.id)
            ).fetchall()

            news_context = []
            for n in news_rows[:5]:
                date_str = f" ({n.date})" if n.date else ""
                news_context.append(f"- {n.headline}{date_str}: {n.snippet[:200]}")

            return {
                "summary": row.summary,
                "industry": row.industry_detected,
                "target_customer": row.target_customer,
                "products_services": row.products_services,
                "tech_stack": row.tech_stack,
                "news": "\n".join(news_context) if news_context else None,
                "source": "database",
            }

    return None


async def _quick_research_company(company_name: str) -> dict:
    """Do a lightweight company research on the fly (website + news)."""
    intel = {
        "summary": None,
        "industry": None,
        "target_customer": None,
        "products_services": None,
        "tech_stack": None,
        "news": None,
        "source": "live_research",
    }

    # Quick website fetch
    slug = re.sub(r"[^a-z0-9]", "", company_name.lower())
    website_url = f"https://www.{slug}.com"

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
        ) as client:
            resp = await client.get(website_url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                for tag in soup.find_all(["nav", "footer", "script", "style", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)[:3000]
                if len(text) > 100:
                    intel["summary"] = text
    except Exception:
        pass

    # Quick news search
    try:
        ddgs = DDGS()
        results = list(ddgs.news(f'"{company_name}" funding OR launch OR hiring', max_results=5))
        if results:
            lines = []
            for r in results[:5]:
                date_str = f" ({r.get('date', '')})" if r.get("date") else ""
                lines.append(f"- {r.get('title', '')}{date_str}: {r.get('body', '')[:200]}")
            intel["news"] = "\n".join(lines)
    except Exception:
        pass

    return intel


# ── LinkedIn message drafting ────────────────────────────────────────────────

LINKEDIN_MESSAGE_PROMPT = """\
You are writing a LinkedIn connection request message on behalf of a representative from Nexa, \
a software architecture and delivery company that helps scaling companies build software predictably.

## Recipient
- **Name**: {person_name}
- **Title**: {person_title}
- **Company**: {company_name}
- **Persona**: {persona_label}

## Company intelligence
{company_summary}

## Recent news / buying triggers
{news_context}

## Persona-specific messaging
- **Their likely pain points**: {pain_points}
- **Nexa's promise for them**: {nexa_promise}
- **Tone**: {tone}

## Instructions
Write a short LinkedIn connection request message (MUST be under 280 characters) that:

1. Opens with a specific, genuine reference to their company or a recent event (from news/intelligence above).
2. Briefly hints at a shared interest or relevant challenge for their role.
3. Ends with a simple, non-pushy ask to connect.

## Constraints
- MAXIMUM 280 characters total (this is a hard LinkedIn limit for connection notes).
- Conversational and authentic — not salesy.
- No buzzwords, no exclamation marks.
- Address them by first name.
- Do NOT mention Nexa by name — just express genuine interest.
- Do NOT use generic openers like "I came across your profile".

Return ONLY the message text, nothing else.
"""


async def draft_linkedin_message(profile: dict, company_intel: dict) -> str:
    """Draft a personalized LinkedIn connection message using Claude."""
    client = Anthropic(api_key=settings.anthropic_api_key)

    persona_key = profile.get("persona", "founder_ceo")
    persona = PERSONA_CONTEXT.get(persona_key, PERSONA_CONTEXT["founder_ceo"])

    # Build company summary
    company_summary = company_intel.get("summary") or "No company details available."
    if company_intel.get("industry"):
        company_summary += f"\nIndustry: {company_intel['industry']}"
    if company_intel.get("products_services"):
        company_summary += f"\nProducts/Services: {company_intel['products_services']}"

    news_context = company_intel.get("news") or "No recent news found."

    prompt = LINKEDIN_MESSAGE_PROMPT.format(
        person_name=profile.get("name") or "there",
        person_title=profile.get("title") or "Unknown",
        company_name=profile.get("company") or "their company",
        persona_label=persona["label"],
        company_summary=company_summary,
        news_context=news_context,
        pain_points=persona["pain_points"],
        nexa_promise=persona["nexa_promise"],
        tone=persona["tone"],
    )

    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()

    # Strip quotes if Claude wrapped the message
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    return text


# ── Main orchestrator ────────────────────────────────────────────────────────

async def qualify_profile(
    linkedin_url: str,
    headless: bool = False,
    skip_message: bool = False,
):
    """Full qualification flow: scrape → evaluate ICP → draft message."""

    console.print(f"\n[bold]LinkedIn Profile Qualification[/bold]")
    console.print(f"URL: {linkedin_url}\n")

    # ── Step 1: Scrape the profile ──
    console.print("[bold cyan]Step 1/3:[/bold cyan] Scraping profile...")

    pw, context = await create_browser(headless=headless)
    try:
        page = context.pages[0] if context.pages else await new_stealth_page(context)

        # Login if needed (for Sales Nav URLs)
        if "/sales/" in linkedin_url:
            await wait_for_linkedin_login(page)
            await random_delay(1, 2)

        profile = await scrape_linkedin_profile(page, linkedin_url)
    finally:
        await context.close()
        await pw.stop()

    if not profile["name"]:
        console.print("[red]Could not extract profile data. The page may require login or the URL may be invalid.[/red]")
        return

    # Display profile info
    profile_table = Table(show_header=False, box=None, pad_edge=False)
    profile_table.add_column("Field", style="bold", min_width=10)
    profile_table.add_column("Value")
    profile_table.add_row("Name", profile["name"] or "—")
    profile_table.add_row("Title", profile["title"] or "—")
    profile_table.add_row("Company", profile["company"] or "—")
    profile_table.add_row("Location", profile["location"] or "—")
    profile_table.add_row("Persona", profile["persona"])
    console.print(Panel(profile_table, title="[bold]Profile[/bold]", border_style="blue"))

    # ── Step 2: Evaluate ICP fit ──
    console.print("\n[bold cyan]Step 2/3:[/bold cyan] Evaluating ICP fit...")

    verdict = check_icp_fit(profile)

    # Display verdict
    score = verdict.get("score", 0)
    qualified = verdict.get("qualified", False)
    score_color = "green" if score >= 70 else "yellow" if score >= 40 else "red"
    status_text = "[bold green]✓ QUALIFIED[/bold green]" if qualified else "[bold red]✗ NOT QUALIFIED[/bold red]"

    verdict_lines = [f"Status: {status_text}", f"Score: [{score_color}]{score}/100[/{score_color}]"]
    if verdict.get("company_industry_guess"):
        verdict_lines.append(f"Industry (guess): {verdict['company_industry_guess']}")
    verdict_lines.append("")

    if verdict.get("reasons"):
        verdict_lines.append("[bold]Reasons:[/bold]")
        for r in verdict["reasons"]:
            verdict_lines.append(f"  [green]✓[/green] {r}")

    if verdict.get("gaps"):
        verdict_lines.append("[bold]Gaps:[/bold]")
        for g in verdict["gaps"]:
            verdict_lines.append(f"  [yellow]![/yellow] {g}")

    console.print(Panel(
        "\n".join(verdict_lines),
        title="[bold]ICP Evaluation[/bold]",
        border_style="green" if qualified else "red",
    ))

    if not qualified:
        console.print("\n[yellow]Profile does not match ICP. Skipping message draft.[/yellow]")
        return

    if skip_message:
        console.print("\n[dim]Message drafting skipped (--skip-message).[/dim]")
        return

    # ── Step 3: Gather intel & draft message ──
    console.print("\n[bold cyan]Step 3/3:[/bold cyan] Gathering company intelligence & drafting message...")

    # Try DB first, then live research
    company_intel = _lookup_company_in_db(profile["company"])
    if company_intel:
        console.print(f"  [green]✓[/green] Found existing research in database")
    else:
        console.print(f"  [dim]No existing research found, doing quick lookup...[/dim]")
        company_intel = await _quick_research_company(profile["company"] or profile["name"])
        if company_intel.get("summary"):
            console.print(f"  [green]✓[/green] Website fetched")
        if company_intel.get("news"):
            console.print(f"  [green]✓[/green] News found")

    message = await draft_linkedin_message(profile, company_intel)

    # Display the drafted message
    char_count = len(message)
    char_color = "green" if char_count <= 280 else "red"

    console.print(Panel(
        message,
        title="[bold]LinkedIn Connection Message[/bold]",
        subtitle=f"[{char_color}]{char_count}/280 chars[/{char_color}]",
        border_style="green",
    ))

    if char_count > 280:
        console.print("[yellow]Warning: Message exceeds 280-character LinkedIn limit. Consider shortening.[/yellow]")

    console.print("\n[green]✓ Qualification complete![/green]")
