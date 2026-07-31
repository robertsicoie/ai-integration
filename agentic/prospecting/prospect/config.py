from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # API keys
    anthropic_api_key: str = ""
    apollo_api_key: str = ""

    # LinkedIn
    linkedin_email: str = ""
    linkedin_password: str = ""

    # Browser
    browser_profile_dir: str = str(Path.home() / ".prospect_browser")

    # Database
    db_path: str = "./prospect_data.db"

    # Rate limits
    linkedin_daily_limit: int = 80
    linkedin_delay_min: float = 2.0
    linkedin_delay_max: float = 6.0

    # Claude
    claude_model: str = "claude-sonnet-4-20250514"

    # ── ICP Configuration ──
    # Company size range (comma-separated, e.g. "11-50,51-200")
    icp_company_sizes: str = "11-50,51-200"

    # Funding stages (comma-separated, e.g. "Seed,Series A,Series B,Series C")
    icp_funding_stages: str = "Seed,Series A,Series B,Series C"

    # Target industries (comma-separated)
    icp_industries: str = "fintech,logistics,marketplace,SaaS,proptech,healthcare technology,betting,gaming,manufacturing"

    # Buyer persona job titles (comma-separated)
    icp_titles: str = "CEO,Founder,Co-Founder,CTO,Chief Technology Officer,VP Engineering,VP of Engineering,Head of Engineering,Head of Product,Product Director,VP Product"

    # Industries to exclude (comma-separated)
    icp_exclude_industries: str = ""

    # Geographic focus (comma-separated country/region names, empty = worldwide)
    icp_locations: str = ""

    def get_company_sizes(self) -> list[str]:
        return [s.strip() for s in self.icp_company_sizes.split(",") if s.strip()]

    def get_funding_stages(self) -> list[str]:
        return [s.strip() for s in self.icp_funding_stages.split(",") if s.strip()]

    def get_industries(self) -> list[str]:
        return [s.strip() for s in self.icp_industries.split(",") if s.strip()]

    def get_titles(self) -> list[str]:
        return [s.strip() for s in self.icp_titles.split(",") if s.strip()]

    def get_exclude_industries(self) -> list[str]:
        return [s.strip() for s in self.icp_exclude_industries.split(",") if s.strip()]

    def get_locations(self) -> list[str]:
        return [s.strip() for s in self.icp_locations.split(",") if s.strip()]


settings = Settings()
