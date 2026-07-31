"""Stage 4: AI-powered personalized email drafting."""

import sqlalchemy as sa
from anthropic import Anthropic
from rich.console import Console
from rich.progress import Progress

from prospect.config import settings
from prospect.db import companies, persons, news_items, draft_emails, init_db

console = Console()

PERSONA_CONTEXT = {
    "founder_ceo": {
        "label": "Founder / CEO",
        "pain_points": (
            "burned by agencies, unclear development timelines, features misunderstood, "
            "budget overruns, need clarity before committing budget"
        ),
        "nexa_promise": (
            "Nexa provides clarity before cost — we map business intent to executable systems "
            "with predictable delivery. No surprises, no scope creep."
        ),
        "tone": "Strategic, ROI-focused, empathetic to founder frustrations",
    },
    "cto": {
        "label": "CTO / VP Engineering",
        "pain_points": (
            "architecture drifting, teams moving fast but inconsistently, backlog chaos, "
            "constant refactoring, technical debt accumulating"
        ),
        "nexa_promise": (
            "Nexa brings architecture governance + agentic acceleration — we help your team "
            "ship consistently on a clean, scalable foundation."
        ),
        "tone": "Technical, peer-to-peer, pragmatic",
    },
    "vp_engineering": {
        "label": "VP / Head of Engineering",
        "pain_points": (
            "architecture drifting, teams moving fast but inconsistently, backlog chaos, "
            "constant refactoring, delivery unpredictability"
        ),
        "nexa_promise": (
            "Nexa brings architecture governance + agentic acceleration — we help your team "
            "ship consistently on a clean, scalable foundation."
        ),
        "tone": "Technical, focused on team productivity and delivery",
    },
    "head_of_product": {
        "label": "Head of Product / Product Director",
        "pain_points": (
            "requirements misunderstood, slow translation from business to engineering, "
            "lack of traceability, features that don't match intent"
        ),
        "nexa_promise": (
            "Nexa translates business intent into executable systems — full traceability "
            "from product vision to shipped features."
        ),
        "tone": "Product-focused, bridge between business and engineering",
    },
}

EMAIL_PROMPT = """\
You are writing a cold outreach email on behalf of Nexa, a software architecture and delivery company. \
Nexa helps scaling companies build software predictably by translating business intent into executable systems.

## Recipient context
- **Name**: {person_name}
- **Title**: {person_title}
- **Company**: {company_name}
- **Persona**: {persona_label}

## Company intelligence
{company_summary}

## Industry
{industry}

## Recent news / buying triggers
{news_context}

## Persona-specific messaging
- **Their likely pain points**: {pain_points}
- **Nexa's promise for them**: {nexa_promise}
- **Tone**: {tone}

## Instructions
Write a short personalized cold email (3-4 paragraphs, under 150 words total) that:

1. **Opens with a specific, genuine observation** about their company based on the intelligence above. \
Reference something concrete — a recent funding round, product launch, hiring surge, or a specific aspect of what they do. \
Do NOT open with generic flattery.

2. **Connects to a relevant pain point** for their specific role. Make it feel like you understand \
the challenges someone in their position faces at this stage of growth. Be subtle, not pushy.

3. **Briefly introduces Nexa's value** in 1-2 sentences. Focus on the outcome, not the process.

4. **Soft CTA**: Ask for a 15-minute conversation, not a demo. Frame it as exploring whether there's a fit.

## Constraints
- Conversational tone, not salesy. No buzzwords like "synergy", "leverage", "cutting-edge".
- No exclamation marks.
- Subject line should be short (under 50 chars), specific, and curiosity-driven.
- Sign off as "Best, [Name]" — leave [Name] as a placeholder.

Return your response in this exact format:
SUBJECT: <subject line>
---
<email body>
"""


def _build_news_context(news: list) -> str:
    """Format news items into context for the prompt."""
    if not news:
        return "No recent news found."

    lines = []
    for item in news[:5]:
        date_str = f" ({item.date})" if item.date else ""
        lines.append(f"- {item.headline}{date_str}: {item.snippet[:200]}")
    return "\n".join(lines)


def _parse_email_response(response: str) -> tuple[str, str]:
    """Parse subject and body from Claude's response."""
    subject = ""
    body = response

    if "SUBJECT:" in response:
        parts = response.split("---", 1)
        subject_line = parts[0].replace("SUBJECT:", "").strip()
        subject = subject_line
        if len(parts) > 1:
            body = parts[1].strip()

    return subject, body


async def draft_outreach_emails(batch_size: int = 10, regenerate: bool = False):
    """Draft personalized outreach emails for all leads."""
    engine = init_db()
    client = Anthropic(api_key=settings.anthropic_api_key)

    console.print("[bold]Stage 4: Email Drafting[/bold]")

    with engine.connect() as conn:
        # Find persons whose companies have been researched
        query = (
            sa.select(
                persons,
                companies.c.name.label("company_name"),
                companies.c.summary.label("company_summary"),
                companies.c.industry_detected,
            )
            .join(companies, persons.c.company_id == companies.c.id)
            .where(companies.c.research_status == "done")
        )

        if not regenerate:
            # Exclude persons who already have drafts
            existing_drafts = sa.select(draft_emails.c.person_id)
            query = query.where(persons.c.id.notin_(existing_drafts))

        leads = conn.execute(query.limit(batch_size)).fetchall()

    if not leads:
        console.print("[yellow]No leads pending email drafts.[/yellow]")
        return

    console.print(f"Drafting emails for {len(leads)} leads...")

    with Progress() as progress:
        task = progress.add_task("Drafting emails...", total=len(leads))

        for lead in leads:
            persona_key = lead.persona or "founder_ceo"
            persona = PERSONA_CONTEXT.get(persona_key, PERSONA_CONTEXT["founder_ceo"])

            # Get news for this company
            with engine.connect() as conn:
                news = conn.execute(
                    sa.select(news_items).where(news_items.c.company_id == lead.company_id)
                ).fetchall()

            news_context = _build_news_context(news)

            try:
                prompt = EMAIL_PROMPT.format(
                    person_name=lead.name,
                    person_title=lead.title,
                    company_name=lead.company_name,
                    persona_label=persona["label"],
                    company_summary=lead.company_summary or "No summary available.",
                    industry=lead.industry_detected or "Unknown",
                    news_context=news_context,
                    pain_points=persona["pain_points"],
                    nexa_promise=persona["nexa_promise"],
                    tone=persona["tone"],
                )

                message = client.messages.create(
                    model=settings.claude_model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )

                response_text = message.content[0].text
                subject, body = _parse_email_response(response_text)

                # Get current max version for this person
                with engine.connect() as conn:
                    max_version = conn.execute(
                        sa.select(sa.func.max(draft_emails.c.version)).where(
                            draft_emails.c.person_id == lead.id
                        )
                    ).scalar() or 0

                    conn.execute(
                        sa.insert(draft_emails).values(
                            person_id=lead.id,
                            company_id=lead.company_id,
                            subject=subject,
                            body=body,
                            persona_template=persona_key,
                            version=max_version + 1,
                        )
                    )
                    conn.commit()

                console.print(
                    f"  [green]✓[/green] {lead.name} ({lead.title} @ {lead.company_name})"
                    f" — Subject: {subject}"
                )

            except Exception as e:
                console.print(f"  [red]✗[/red] {lead.name}: {e}")

            progress.advance(task)

    console.print(f"\n[green]✓ Email drafting complete![/green]")
