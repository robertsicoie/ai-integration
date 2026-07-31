# Prospect — B2B Prospecting Pipeline

Automated prospecting pipeline that finds companies matching your ICP on LinkedIn Sales Navigator, researches them, finds buying triggers in the press, and drafts personalized outreach emails using Claude.

## Pipeline Stages

1. **LinkedIn Scrape** — Search Sales Navigator for companies (10–200 employees, Series A–C, target industries) and extract decision makers (CEO, CTO, VP Eng, Head of Product)
2. **Company Research** — Visit each company's website, summarize what they do, detect industry and tech stack using Claude
3. **News Research** — Search for buying triggers: funding rounds, product launches, hiring surges, legacy replacements
4. **Email Drafting** — Generate personalized outreach emails matched to each persona's pain points

Each stage is independently runnable and resumable. Progress is tracked in a local SQLite database.

## Installation

### Prerequisites

- Python 3.11+
- A LinkedIn Sales Navigator account
- An [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
# Clone and navigate to the project
cd prospecting

# Install the package and dependencies
pip install -e .

# Install the Playwright browser
playwright install chromium

# Copy and fill in your credentials
cp .env.example .env
```

Edit `.env` with your credentials:

```
ANTHROPIC_API_KEY=sk-ant-...
LINKEDIN_EMAIL=your@email.com
LINKEDIN_PASSWORD=your-password
```

### Initialize the database

```bash
prospect init
```

## Usage

### Run the full pipeline

```bash
prospect run-all
```

This runs all 4 stages sequentially. On first run, a browser window opens for you to log into LinkedIn Sales Navigator manually. The session is saved for future runs.

### Run stages individually

```bash
# Stage 1: Scrape LinkedIn (opens browser)
prospect scrape --max-results 200

# Stage 2: Research company websites
prospect research --batch-size 10

# Stage 3: Search news and press
prospect news --batch-size 10

# Stage 4: Draft outreach emails
prospect draft --batch-size 10
```

### Check progress

```bash
prospect status
```

Outputs a table showing pending/done/error counts for each stage.

### Export results

```bash
# Export to CSV (default)
prospect export

# Export to JSON
prospect export --format json

# Custom output filename
prospect export --output my_prospects
```

### Regenerate emails

```bash
prospect draft --regenerate
```

Creates new email versions for leads that already have drafts, keeping previous versions in the database.

## Configuration

All settings are configured via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Claude API key |
| `LINKEDIN_EMAIL` | — | LinkedIn login email |
| `LINKEDIN_PASSWORD` | — | LinkedIn login password |
| `BROWSER_PROFILE_DIR` | `~/.prospect_browser` | Persistent browser session directory |
| `DB_PATH` | `./prospect_data.db` | SQLite database path |
| `LINKEDIN_DAILY_LIMIT` | `80` | Max LinkedIn actions per day |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model for summarization and emails |

## ICP Filters

The pipeline targets:

- **Company size**: 10–200 employees
- **Funding**: Series A through Series C
- **Industries**: fintech, logistics, marketplaces, SaaS, proptech, healthtech, betting/gaming, manufacturing platforms
- **Buyer personas**: Founder/CEO, CTO, VP Engineering, Head of Product

## Safety

- **Rate limiting**: 80 LinkedIn actions/day by default, with human-like delays (2–6s between actions). Limits persist across restarts.
- **Anti-detection**: Stealth browser patches, realistic user agent, non-headless mode by default.
- **Resumable**: Every record tracks its status. Stop anytime and pick up where you left off.

## Project Structure

```
prospect/
├── cli.py                  # Typer CLI entry point
├── config.py               # Settings from .env
├── db.py                   # SQLite schema and connections
├── models.py               # Pydantic data models
├── rate_limiter.py         # Daily rate limiter (persisted)
├── browser.py              # Playwright stealth browser setup
└── stages/
    ├── linkedin.py         # Stage 1: Sales Navigator scraping
    ├── company_research.py # Stage 2: Website research + Claude
    ├── news_research.py    # Stage 3: DuckDuckGo news search
    └── email_draft.py      # Stage 4: Claude email drafting
```
