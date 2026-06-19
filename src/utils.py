# ==========================================================
# utils.py
# Common Utility Functions
# ==========================================================

import re
from urllib.parse import urlparse, unquote


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# EXTRACT URL SLUG
# ==========================================================

def slug_text(url):

    path = urlparse(url).path

    slug = path.split("/")[-1]

    slug = unquote(slug)

    slug = slug.replace("-", " ")
    slug = slug.replace("_", " ")

    return slug.strip()


# ==========================================================
# SAVE DATAFRAME TO CSV
# ==========================================================

def save_csv(df, filepath):

    df.to_csv(
        filepath,
        index=False
    )

    print(
        f"Saved successfully -> {filepath}"
    )