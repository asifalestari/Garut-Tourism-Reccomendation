import logging
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from config import settings

logger = logging.getLogger("pipeline")

class BrowserManager:
    """
    Context manager to handle Playwright browser lifetime, context creation,
    and page initialization cleanly.
    """
    def __init__(self) -> None:
        self._playwright = None
        self._browser: Browser = None
        self._context: BrowserContext = None
        self._page: Page = None

    def __enter__(self):
        logger.info("Initializing Playwright browser...")

        self._playwright = sync_playwright().start()

        # Launch standard Playwright Chromium browser
        self._browser = self._playwright.chromium.launch(
            headless=settings.HEADLESS
        )

        # Create standard context using settings configurations
        self._context = self._browser.new_context(
            locale=settings.LOCALE,
            viewport=settings.VIEWPORT
        )

        return self

    def get_page(self) -> Page:
        """
        Creates and returns a single Page inside the active context.
        Reuses the page if already created.
        """
        if not self._context:
            raise RuntimeError("BrowserManager must be used as a context manager.")
        
        if self._page is None:
            logger.info("Opening new page/tab...")
            self._page = self._context.new_page()

        return self._page

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Closing Playwright...")

        if self._page:
            try:
                self._page.close()
            except Exception:
                pass

        if self._context:
            try:
                self._context.close()
            except Exception:
                pass

        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass

        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass

        logger.info("Playwright shutdown complete.")
