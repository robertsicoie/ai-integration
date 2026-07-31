"""Stage 1: LinkedIn Sales Navigator scraping.

Strategy: Navigate to Sales Navigator account search, apply filters via the UI,
then extract results. LinkedIn Sales Navigator uses dynamic SPAs with frequently
changing selectors, so we use broad fallback strategies and screenshot on errors.
"""

import asyncio
import re
import hashlib
from pathlib import Path
from urllib.parse import quote

import sqlalchemy as sa
from rich.console import Console
from rich.progress import Progress

from prospect.browser import (
    create_browser,
    human_scroll,
    new_stealth_page,
    random_delay,
    wait_for_linkedin_login,
)
from prospect.config import settings
from prospect.db import companies, persons, init_db
from prospect.rate_limiter import RateLimiter

console = Console()

def _get_buyer_titles() -> list[str]:
    """Get buyer titles from ICP config."""
    return settings.get_titles() or [
        "CEO", "Founder", "Co-Founder", "CTO",
        "VP Engineering", "VP of Engineering", "Head of Engineering",
        "Head of Product", "Product Director", "VP Product",
    ]

DEBUG_DIR = Path("./debug_screenshots")


def _make_id(name: str) -> str:
    """Create a stable ID from a name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    short_hash = hashlib.md5(name.encode()).hexdigest()[:6]
    return f"{slug}-{short_hash}"


def _map_persona(title: str) -> str:
    """Map a job title to a buyer persona."""
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


async def _screenshot_debug(page, name: str):
    """Save a debug screenshot."""
    DEBUG_DIR.mkdir(exist_ok=True)
    path = DEBUG_DIR / f"{name}.png"
    try:
        await page.screenshot(path=str(path))
        console.print(f"  [dim]Debug screenshot: {path}[/dim]")
    except Exception:
        pass


async def _extract_accounts_from_page(page) -> list[dict]:
    """Extract company/account data from Sales Navigator account search results.

    Uses multiple selector strategies since LinkedIn frequently changes the DOM.
    """
    results = []

    # Wait for the page to load results
    await asyncio.sleep(3)
    await human_scroll(page, times=3)
    await asyncio.sleep(1)

    # Get the full page HTML to analyze structure
    html = await page.content()

    # Strategy 1: Look for result cards by common Sales Navigator patterns
    # Sales Nav account results typically have links to /sales/company/ paths
    card_selectors = [
        # Account search result items
        "li.artdeco-list__item",
        "[data-view-name='search-results-entity']",
        "ol > li[class*='artdeco']",
        "div[class*='search-results'] li",
        "main li[class*='list__item']",
    ]

    items = []
    for sel in card_selectors:
        items = await page.query_selector_all(sel)
        if items:
            console.print(f"  [dim]Found {len(items)} items with selector: {sel}[/dim]")
            break

    if not items:
        # Fallback: try to find any result-like containers
        items = await page.query_selector_all("li")
        # Filter to only those containing company links
        filtered = []
        for item in items:
            link = await item.query_selector("a[href*='/sales/company/'], a[href*='/sales/lead/']")
            if link:
                filtered.append(item)
        items = filtered
        if items:
            console.print(f"  [dim]Found {len(items)} items via link fallback[/dim]")

    if not items:
        await _screenshot_debug(page, "no_results")
        console.print("[yellow]  Could not find result items. Screenshot saved for debugging.[/yellow]")
        return results

    for item in items:
        try:
            # Extract company name — look for the primary link text
            name = None
            linkedin_url = None
            headcount = None
            industry = None

            # Find company name from links
            name_link_selectors = [
                "a[data-anonymize='company-name']",
                "a[href*='/sales/company/']",
                "a[data-control-name*='company']",
                "span[data-anonymize='company-name']",
            ]
            for sel in name_link_selectors:
                el = await item.query_selector(sel)
                if el:
                    name = (await el.inner_text()).strip()
                    href = await el.get_attribute("href") or ""
                    if href:
                        linkedin_url = href if href.startswith("http") else f"https://www.linkedin.com{href}"
                    break

            if not name:
                continue

            # Extract headcount
            headcount_selectors = [
                "[data-anonymize='headcount']",
                "span[class*='company-size']",
                "span[class*='headcount']",
            ]
            for sel in headcount_selectors:
                el = await item.query_selector(sel)
                if el:
                    headcount = (await el.inner_text()).strip()
                    break

            # If no specific headcount element, try to find it in item text
            if not headcount:
                item_text = await item.inner_text()
                # Look for patterns like "11-50" or "51-200"
                match = re.search(r"(\d+[-–]\d+)\s*employees?", item_text, re.IGNORECASE)
                if match:
                    headcount = match.group(1)

            # Extract industry
            industry_selectors = [
                "[data-anonymize='industry']",
                "span[class*='industry']",
            ]
            for sel in industry_selectors:
                el = await item.query_selector(sel)
                if el:
                    industry = (await el.inner_text()).strip()
                    break

            results.append({
                "id": _make_id(name),
                "name": name,
                "linkedin_url": linkedin_url,
                "size_range": headcount,
                "industry": industry,
                "scrape_status": "done",
                "research_status": "pending",
                "news_status": "pending",
            })

        except Exception as e:
            console.print(f"  [dim]Skipping item: {e}[/dim]")
            continue

    return results


async def _extract_leads_from_page(page) -> list[dict]:
    """Extract lead/person data from Sales Navigator lead search results."""
    results = []

    await asyncio.sleep(3)
    try:
        await human_scroll(page, times=2)
    except Exception:
        pass  # Page might have navigated or closed
    await asyncio.sleep(1)

    # Find lead cards
    items = await page.query_selector_all("li.artdeco-list__item")
    if not items:
        items = await page.query_selector_all("ol > li")

    for item in items:
        try:
            name = None
            linkedin_url = None
            title = None

            # Person name — Sales Nav uses span[data-anonymize='person-name']
            name_el = await item.query_selector("span[data-anonymize='person-name']")
            if name_el:
                name = (await name_el.inner_text()).strip()

            # Profile link — look for the link with profile-link in data attribute
            link_el = await item.query_selector("a[data-lead-search-result*='profile-link']")
            if not link_el:
                # Fallback: any link to /sales/lead/
                link_el = await item.query_selector("a[href*='/sales/lead/']")
            if link_el:
                href = await link_el.get_attribute("href") or ""
                if href:
                    linkedin_url = href if href.startswith("http") else f"https://www.linkedin.com{href}"
                if not name:
                    name = (await link_el.inner_text()).strip()

            if not name:
                continue

            # Title — span[data-anonymize='title']
            title_el = await item.query_selector("span[data-anonymize='title']")
            if title_el:
                title = (await title_el.inner_text()).strip()

            if not title:
                # Fallback: parse from item text
                item_text = await item.inner_text()
                lines = [l.strip() for l in item_text.split("\n") if l.strip()]
                for i, line in enumerate(lines):
                    if name and name in line and i + 1 < len(lines):
                        title = lines[i + 1]
                        break

            results.append({
                "name": name,
                "title": title or "Unknown",
                "linkedin_url": linkedin_url,
            })

        except Exception as e:
            console.print(f"  [dim]Skipping lead: {e}[/dim]")
            continue

    return results


async def _extract_email_from_profile(page, profile_url: str) -> str | None:
    """Visit a lead's Sales Navigator profile and try to extract their email.

    Sales Navigator shows emails in the 'Contact information' section
    for 1st-degree connections or when email credits have been used.
    """
    try:
        await page.goto(profile_url)
        await asyncio.sleep(3)

        # 1. Check for mailto links
        mailto = await page.query_selector("a[href^='mailto:']")
        if mailto:
            href = await mailto.get_attribute("href") or ""
            if href.startswith("mailto:"):
                return href.replace("mailto:", "").split("?")[0].strip()

        # 2. Check for email in contact info section
        # Sales Nav shows email with data-anonymize="email" in some layouts
        email_el = await page.query_selector("[data-anonymize='email']")
        if email_el:
            email_text = (await email_el.inner_text()).strip()
            if "@" in email_text:
                return email_text

        # 3. Scan visible text for email patterns
        # Look specifically in the right rail / contact info area
        selectors_to_check = [
            "aside",
            "[class*='right-rail']",
            "[class*='contact']",
            "[class*='profile-topcard']",
        ]
        for sel in selectors_to_check:
            try:
                el = await page.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
                    if emails:
                        return emails[0]
            except Exception:
                continue

    except Exception:
        pass

    return None


async def _has_next_page(page) -> bool:
    """Check if there's a next page button and click it."""
    next_selectors = [
        "button[aria-label='Next']",
        "button.artdeco-pagination__button--next",
        "[class*='pagination'] button:last-child",
    ]
    for sel in next_selectors:
        btn = await page.query_selector(sel)
        if btn and await btn.is_enabled():
            await btn.click()
            await asyncio.sleep(2)
            return True
    return False


async def scrape_linkedin(max_results: int = 200, headless: bool = False):
    """Main entry point for LinkedIn Sales Navigator scraping."""
    engine = init_db()
    limiter = RateLimiter(
        action_type="linkedin_search",
        max_per_day=settings.linkedin_daily_limit,
        delay_min=settings.linkedin_delay_min,
        delay_max=settings.linkedin_delay_max,
    )

    console.print("[bold]Stage 1: LinkedIn Sales Navigator Scraping[/bold]")
    console.print(f"Daily limit: {limiter.remaining()}/{settings.linkedin_daily_limit} remaining")

    pw, context = await create_browser(headless=headless)

    try:
        page = context.pages[0] if context.pages else await new_stealth_page(context)

        # Ensure we're logged in
        await wait_for_linkedin_login(page)
        await random_delay(2, 4)

        # ── Phase 1: Search for accounts (companies) ──
        console.print("\n[bold]Phase 1: Searching for companies...[/bold]")

        # Build account search URL from ICP config
        # LinkedIn Sales Nav headcount codes: A=1-10, B=11-50, C=51-200, D=201-500, E=501-1000
        size_code_map = {
            "1-10": ("A", "1-10"), "11-50": ("B", "11-50"),
            "51-200": ("C", "51-200"), "201-500": ("D", "201-500"),
            "501-1000": ("E", "501-1000"),
        }
        headcount_values = []
        for size in settings.get_company_sizes():
            if size in size_code_map:
                code, text = size_code_map[size]
                headcount_values.append(f"(id:{code},text:{text},selectionType:INCLUDED)")

        if not headcount_values:
            headcount_values = [
                "(id:B,text:11-50,selectionType:INCLUDED)",
                "(id:C,text:51-200,selectionType:INCLUDED)",
            ]

        account_search_url = (
            "https://www.linkedin.com/sales/search/company"
            "?query=(filters:List("
            "(type:COMPANY_HEADCOUNT,values:List("
            + ",".join(headcount_values) +
            "))"
            "))"
        )
        console.print(f"  Navigating to account search...")
        await page.goto(account_search_url)
        await random_delay(3, 5)

        # Clear any sticky/pinned filters from previous sessions
        # The screenshot shows Sales Nav keeps "United States" and "Professional Services"
        try:
            clear_btn = await page.query_selector("button:has-text('Clear all')")
            if clear_btn and await clear_btn.is_visible():
                await clear_btn.click()
                console.print("  [dim]Cleared pinned filters[/dim]")
                await random_delay(2, 3)
                # Re-navigate with our filters after clearing
                await page.goto(account_search_url)
                await random_delay(3, 5)
        except Exception:
            pass

        # Take a debug screenshot to see what loaded
        await _screenshot_debug(page, "account_search_loaded")

        total_companies = 0
        page_num = 1

        with Progress() as progress:
            task = progress.add_task("Scraping companies...", total=max_results)

            while total_companies < max_results and limiter.can_acquire():
                await limiter.wait_and_acquire()

                company_data = await _extract_accounts_from_page(page)

                if not company_data:
                    # Try one more time after waiting
                    await random_delay(3, 5)
                    company_data = await _extract_accounts_from_page(page)

                if not company_data:
                    console.print("[yellow]No more results found, stopping.[/yellow]")
                    break

                # Deduplicate and store
                new_count = 0
                with engine.connect() as conn:
                    for c in company_data:
                        existing = conn.execute(
                            sa.select(companies.c.id).where(companies.c.id == c["id"])
                        ).first()
                        if not existing:
                            conn.execute(sa.insert(companies).values(**c))
                            new_count += 1
                    conn.commit()

                total_companies += new_count
                progress.update(task, completed=total_companies)
                console.print(f"  Page {page_num}: {len(company_data)} found, {new_count} new (total: {total_companies})")

                # Go to next page
                if not await _has_next_page(page):
                    console.print("  [yellow]No more pages.[/yellow]")
                    break

                page_num += 1
                await random_delay(2, 4)

        # ── Phase 2: Find decision makers for each company ──
        console.print(f"\n[bold]Phase 2: Finding decision makers...[/bold]")

        with engine.connect() as conn:
            company_rows = conn.execute(
                sa.select(companies.c.id, companies.c.linkedin_url, companies.c.name).where(
                    companies.c.scrape_status == "done"
                )
            ).fetchall()

        total_leads = 0

        with Progress() as progress:
            task = progress.add_task("Finding leads...", total=len(company_rows))

            for row in company_rows:
                if not limiter.can_acquire():
                    console.print("[yellow]Daily limit reached. Resume tomorrow.[/yellow]")
                    break

                company_name = row.name
                linkedin_url = row.linkedin_url or ""
                console.print(f"  Searching leads at {company_name}...")

                # Extract LinkedIn company ID from URL like /sales/company/2864?...
                company_li_id = ""
                if "/sales/company/" in linkedin_url:
                    match = re.search(r"/sales/company/(\d+)", linkedin_url)
                    if match:
                        company_li_id = match.group(1)

                # Build Sales Navigator people search URL with company filter
                # The format uses encoded filter list params
                if company_li_id:
                    lead_search_url = (
                        f"https://www.linkedin.com/sales/search/people"
                        f"?query=(filters:List("
                        f"(type:CURRENT_COMPANY,values:List((id:{company_li_id},selectionType:INCLUDED)))"
                        f"))"
                    )
                else:
                    # Fallback: keyword search with company name
                    lead_search_url = (
                        f"https://www.linkedin.com/sales/search/people"
                        f"?query=(keywords:{quote(company_name)})"
                    )

                try:
                    await page.goto(lead_search_url)
                    await limiter.wait_and_acquire()
                    await random_delay(3, 5)

                    lead_data = await _extract_leads_from_page(page)

                    # Filter to only buyer personas
                    relevant_leads = []
                    for lead in lead_data:
                        persona = _map_persona(lead["title"])
                        if persona != "other":
                            lead["persona"] = persona
                            lead["id"] = _make_id(f"{lead['name']}-{row.id}")
                            lead["company_id"] = row.id
                            relevant_leads.append(lead)

                    with engine.connect() as conn:
                        for lead in relevant_leads:
                            existing = conn.execute(
                                sa.select(persons.c.id).where(persons.c.id == lead["id"])
                            ).first()
                            if not existing:
                                conn.execute(sa.insert(persons).values(**lead))
                                total_leads += 1
                        conn.commit()

                    if relevant_leads:
                        console.print(f"    [green]Found {len(relevant_leads)} decision makers[/green]")
                    else:
                        console.print(f"    [dim]No matching leads found[/dim]")

                except Exception as e:
                    console.print(f"    [red]Error searching leads: {e}[/red]")

                progress.advance(task)

        # ── Phase 3: Extract emails from lead profiles ──
        console.print(f"\n[bold]Phase 3: Extracting contact emails...[/bold]")

        with engine.connect() as conn:
            leads_without_email = conn.execute(
                sa.select(persons.c.id, persons.c.name, persons.c.linkedin_url).where(
                    persons.c.linkedin_url.isnot(None),
                    persons.c.linkedin_url != "",
                    sa.or_(persons.c.email.is_(None), persons.c.email == ""),
                )
            ).fetchall()

        emails_found = 0

        if leads_without_email:
            with Progress() as progress:
                task = progress.add_task("Extracting emails...", total=len(leads_without_email))

                for lead in leads_without_email:
                    if not limiter.can_acquire():
                        console.print("[yellow]Daily limit reached. Resume tomorrow.[/yellow]")
                        break

                    profile_url = lead.linkedin_url
                    if not profile_url.startswith("http"):
                        profile_url = f"https://www.linkedin.com{profile_url}"

                    await limiter.wait_and_acquire()
                    email = await _extract_email_from_profile(page, profile_url)

                    if email:
                        with engine.connect() as conn:
                            conn.execute(
                                sa.update(persons)
                                .where(persons.c.id == lead.id)
                                .values(email=email)
                            )
                            conn.commit()
                        emails_found += 1
                        console.print(f"    [green]✓[/green] {lead.name}: {email}")

                    progress.advance(task)

        console.print(f"\n[green]✓ Done![/green] {total_companies} companies, {total_leads} leads, {emails_found} emails found")
        if total_leads > 0 and emails_found == 0:
            console.print("[dim]  Tip: LinkedIn only shows emails for 1st-degree connections.[/dim]")
            console.print("[dim]  Use Apollo.io (--source apollo) for verified email addresses.[/dim]")

    finally:
        await context.close()
        await pw.stop()
