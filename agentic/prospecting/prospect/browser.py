"""Playwright browser setup with stealth and persistent context."""

import asyncio
import random
from pathlib import Path

from playwright.async_api import async_playwright, BrowserContext, Page, Frame
from playwright_stealth import Stealth

from prospect.config import settings

_stealth = Stealth()
DEBUG_DIR = Path("./debug_screenshots")


async def create_browser(headless: bool = False) -> tuple:
    """Create a persistent browser context with stealth patches.

    Returns (playwright_instance, browser_context) — caller must close both.
    """
    pw = await async_playwright().start()

    context = await pw.chromium.launch_persistent_context(
        user_data_dir=settings.browser_profile_dir,
        headless=headless,
        viewport={"width": 1366, "height": 768},
        locale="en-US",
        timezone_id="America/New_York",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        args=[
            "--disable-blink-features=AutomationControlled",
        ],
    )

    # Apply stealth to the default page
    if context.pages:
        await _stealth.apply_stealth_async(context.pages[0])

    return pw, context


async def new_stealth_page(context: BrowserContext) -> Page:
    """Create a new page with stealth patches applied."""
    page = await context.new_page()
    await _stealth.apply_stealth_async(page)
    return page


async def human_type_into(frame_or_page, selector: str, text: str):
    """Type text into an element with human-like delays."""
    el = await frame_or_page.wait_for_selector(selector, timeout=5000)
    await el.click()
    await asyncio.sleep(0.3)
    # Clear existing content
    await el.fill("")
    for char in text:
        await el.press(f"Key{char.upper()}" if char.isalpha() else char, delay=random.randint(50, 150)) if False else None
        await frame_or_page.keyboard.type(char, delay=random.randint(50, 150))
        if random.random() < 0.05:
            await asyncio.sleep(random.uniform(0.2, 0.5))


async def human_scroll(page: Page, times: int = 3):
    """Scroll down the page like a human."""
    for _ in range(times):
        scroll_amount = random.randint(200, 600)
        await page.mouse.wheel(0, scroll_amount)
        await asyncio.sleep(random.uniform(0.5, 1.5))


async def random_delay(min_s: float = 1.0, max_s: float = 3.0):
    """Sleep a random human-like duration."""
    await asyncio.sleep(random.uniform(min_s, max_s))


def _find_login_frame(page: Page) -> Frame | None:
    """Find the login iframe on Sales Navigator login page."""
    for frame in page.frames:
        if "/uas/login" in frame.url:
            return frame
    return None


async def wait_for_linkedin_login(page: Page):
    """Navigate to Sales Navigator and handle login automatically."""
    await page.goto("https://www.linkedin.com/sales/homepage")
    await asyncio.sleep(4)

    url = page.url

    # Already logged in
    if "/sales/" in url and "login" not in url and "checkpoint" not in url:
        print("✓ Already logged in to Sales Navigator")
        return

    # On login page
    if "login" not in url and "checkpoint" not in url:
        print("✓ Already logged in to Sales Navigator")
        return

    print("Logging in to LinkedIn...")

    email = settings.linkedin_email
    password = settings.linkedin_password

    if not email or not password:
        print("\n⚠️  No LinkedIn credentials in .env")
        print("Please log in manually in the browser window.")
        print("Waiting for login to complete...\n")
        try:
            await page.wait_for_url("**/sales/**", timeout=300_000)
            print("✓ Login successful!")
        except Exception:
            raise RuntimeError("Login timeout — please try again.")
        return

    # Sales Navigator login page embeds the form in an iframe at /uas/login
    # We need to find that iframe and interact with its elements
    login_frame = _find_login_frame(page)

    if login_frame:
        print("  Found login iframe, filling credentials...")
        target = login_frame
    else:
        print("  No iframe found, trying main page...")
        target = page

    # Fill email — #username is the standard ID in both iframe and direct forms
    # Note: Frame objects don't have .keyboard, so we use .fill() for iframes
    try:
        username_el = await target.wait_for_selector("#username", timeout=5000)
        await username_el.click()
        await username_el.fill(email)
        print("  ✓ Email filled")
    except Exception as e:
        print(f"  ⚠️ Could not fill email (#username): {e}")
        # Fallback: try any visible email/text input
        try:
            inputs = await target.query_selector_all("input")
            for inp in inputs:
                inp_type = (await inp.get_attribute("type") or "text").lower()
                if inp_type in ("email", "text", "tel") and await inp.is_visible():
                    await inp.click()
                    await inp.fill(email)
                    print("  ✓ Email filled (fallback)")
                    break
        except Exception:
            print("  ⚠️ Could not find email field. Please log in manually.")
            await page.wait_for_url("**/sales/**", timeout=300_000)
            return

    await random_delay(0.5, 1.0)

    # Fill password
    try:
        password_el = await target.wait_for_selector("#password", timeout=5000)
        await password_el.click()
        await password_el.fill(password)
        print("  ✓ Password filled")
    except Exception as e:
        print(f"  ⚠️ Could not fill password (#password): {e}")
        try:
            password_el = await target.wait_for_selector("input[type='password']", timeout=5000)
            await password_el.click()
            await password_el.fill(password)
            print("  ✓ Password filled (fallback)")
        except Exception:
            print("  ⚠️ Could not find password field. Please log in manually.")
            await page.wait_for_url("**/sales/**", timeout=300_000)
            return

    await random_delay(0.5, 1.0)

    # Click Sign in button
    try:
        btn = await target.wait_for_selector(
            "button[data-litms-control-urn='login-submit']", timeout=5000
        )
        await btn.click()
        print("  ✓ Sign in clicked")
    except Exception:
        # Fallback: submit button
        try:
            btn = await target.wait_for_selector("button[type='submit']", timeout=3000)
            await btn.click()
            print("  ✓ Sign in clicked (fallback)")
        except Exception:
            await target.keyboard.press("Enter")
            print("  ✓ Enter pressed (fallback)")

    # Wait for navigation
    print("  Waiting for login to complete...")
    await asyncio.sleep(5)

    # Check for verification/checkpoint
    if "checkpoint" in page.url or "challenge" in page.url:
        print("\n⚠️  LinkedIn verification required (CAPTCHA or 2FA).")
        print("Please complete the verification in the browser window.")
        print("Waiting...\n")
        try:
            await page.wait_for_url("**/sales/**", timeout=300_000)
        except Exception:
            await page.goto("https://www.linkedin.com/sales/homepage")
            await asyncio.sleep(3)

    # If we landed on regular LinkedIn, navigate to Sales Navigator
    if "/sales/" not in page.url or "login" in page.url:
        await page.goto("https://www.linkedin.com/sales/homepage")
        await asyncio.sleep(5)

    if "/sales/" in page.url and "login" not in page.url:
        print("✓ Login successful!")
    else:
        print(f"⚠️  Not on Sales Navigator yet. Current URL: {page.url}")
        print("  Waiting for manual navigation...")
        await page.wait_for_url("**/sales/**", timeout=300_000)
        print("✓ Login successful!")

    await asyncio.sleep(1)
