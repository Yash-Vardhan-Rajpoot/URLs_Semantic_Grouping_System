# ==========================================================
# scraper.py
# URL Content Extraction Module
# ==========================================================

import requests
import trafilatura

from bs4 import BeautifulSoup
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor

from src.utils import (
    clean_text,
    slug_text
)

# ==========================================================
# HEADERS
# ==========================================================

HEADERS = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


# ==========================================================
# TRAFILATURA EXTRACTION
# ==========================================================

def extract_trafilatura(url):

    try:

        downloaded = trafilatura.fetch_url(url)

        if downloaded:

            text = trafilatura.extract(
                downloaded,
                include_links=False,
                include_comments=False
            )

            if text and len(text) > 200:

                return clean_text(text)

    except:
        pass

    return None


# ==========================================================
# NEWSPAPER EXTRACTION
# ==========================================================

def extract_newspaper(url):

    try:

        article = Article(url)

        article.download()
        article.parse()

        text = (
            article.title +
            " " +
            article.text
        )

        if len(text) > 200:

            return clean_text(text)

    except:
        pass

    return None


# ==========================================================
# BS4 EXTRACTION
# ==========================================================

def extract_bs4(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside"
            ]
        ):
            tag.decompose()

        # ----------------------
        # TITLE
        # ----------------------

        title = ""

        if soup.title:

            title = soup.title.get_text(
                " ",
                strip=True
            )

        # ----------------------
        # META DESCRIPTION
        # ----------------------

        meta_description = ""

        meta = soup.find(
            "meta",
            attrs={"name": "description"}
        )

        if (
            meta and
            meta.get("content")
        ):

            meta_description = meta.get(
                "content"
            )

        # ----------------------
        # HEADINGS
        # ----------------------

        headings = []

        for tag in [
            "h1",
            "h2",
            "h3"
        ]:

            headings.extend(
                [
                    h.get_text(
                        " ",
                        strip=True
                    )
                    for h in soup.find_all(tag)
                ]
            )

        headings = " ".join(
            headings
        )

        # ----------------------
        # BODY CONTENT
        # ----------------------

        body = soup.get_text(
            separator=" ",
            strip=True
        )

        body_words = body.split()

        body = " ".join(
            body_words[:300]
        )

        slug = slug_text(url)

        final_text = f"""
        TITLE:
        {title}

        META:
        {meta_description}

        HEADINGS:
        {headings}

        URL:
        {slug}

        CONTENT:
        {body}
        """

        return clean_text(
            final_text
        )

    except:

        return None


# ==========================================================
# MASTER EXTRACTION
# ==========================================================

def extract_content(url):

    text = extract_trafilatura(url)

    if text:
        return text

    text = extract_newspaper(url)

    if text:
        return text

    text = extract_bs4(url)

    if text:
        return text

    return slug_text(url)


# ==========================================================
# PARALLEL URL SCRAPING
# ==========================================================

def scrape_urls(urls):

    with ThreadPoolExecutor(
        max_workers=20
    ) as executor:

        documents = list(
            executor.map(
                extract_content,
                urls
            )
        )

    return documents