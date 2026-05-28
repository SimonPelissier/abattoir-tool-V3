"""
Slaughterhouse data extraction pipeline.

This module contains the pure-logic pipeline (no Streamlit calls), extracted
from the original Colab notebook. The Streamlit app (app.py) calls these
functions and handles all UI concerns separately.

Architecture
------------
1. generate_queries(company, country)              -> list[str]
2. serpapi_search(queries, gl, hl, location)       -> list[dict]
3. score_links(links, company, country)            -> list[dict]
4. preflight_check(links)                          -> list[dict] (problematic)
5. detect_format(url)                              -> 'PDF'|'HTML'|'CSV'|'EXCEL'
6. extract_content(url, local_path=None)           -> (chunks, fmt, n_chars)
7. extract_abattoirs_from_source(item, company)    -> dict
8. synthesize(all_abattoirs, sources, company)     -> list[dict]
9. geocode_google_first(facility)                  -> updates facility in place
10. refine_capacity(facility, gl, hl, location)    -> updates facility in place
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import time
from typing import Any, Callable, Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash-lite"
CHUNK_SIZE = 2000
OVERLAP = 100
MAX_CHARS_PER_CHUNK = 6000

EXCLUDED_DOMAINS = [
    "yelp.com", "maps.apple.com", "mapquest.com",
    "facebook.com", "instagram.com", "twitter.com",
    "linkedin.com", "tripadvisor.com", "yellowpages.com",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# These will be set by the app at startup
CACHE_DIR: Optional[str] = None
GEMINI_CLIENT = None            # set by configure()
SERPAPI_API_KEY: Optional[str] = None
GOOGLE_PLACES_API_KEY: Optional[str] = None


def configure(
    *,
    cache_dir: str,
    gemini_api_key: str,
    serpapi_key: str,
    google_places_key: str,
) -> None:
    """Initialise the pipeline globals. Called once by the app at startup."""
    global CACHE_DIR, GEMINI_CLIENT, SERPAPI_API_KEY, GOOGLE_PLACES_API_KEY
    from google import genai

    CACHE_DIR = cache_dir
    os.makedirs(CACHE_DIR, exist_ok=True)
    GEMINI_CLIENT = genai.Client(api_key=gemini_api_key)
    SERPAPI_API_KEY = serpapi_key
    GOOGLE_PLACES_API_KEY = google_places_key


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def cached(key: str, func: Callable[[], Any], format: str = "pickle") -> Any:
    """Disk cache. Returns existing value if present, else calls func() and saves."""
    if CACHE_DIR is None:
        return func()
    h = _cache_key(key)
    ext = "json" if format == "json" else "pkl"
    path = os.path.join(CACHE_DIR, f"{h}.{ext}")

    if os.path.exists(path):
        try:
            if format == "json":
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            os.remove(path)

    result = func()
    try:
        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(path, "wb") as f:
                pickle.dump(result, f)
    except Exception:
        pass
    return result


def cache_stats() -> dict:
    """Return basic stats about the cache directory."""
    if not CACHE_DIR or not os.path.isdir(CACHE_DIR):
        return {"files": 0, "size_mb": 0.0}
    files = os.listdir(CACHE_DIR)
    size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
    return {"files": len(files), "size_mb": round(size / 1024 / 1024, 2)}


def cache_clear() -> int:
    """Delete every entry in the cache. Returns the count of removed files."""
    if not CACHE_DIR or not os.path.isdir(CACHE_DIR):
        return 0
    n = 0
    for f in os.listdir(CACHE_DIR):
        try:
            os.remove(os.path.join(CACHE_DIR, f))
            n += 1
        except OSError:
            pass
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Gemini wrapper
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini_json(prompt: str, max_retries: int = 3, retry_wait: int = 15) -> Any:
    """Call Gemini with JSON mime-type, retry on transient errors."""
    from google.genai import types

    for attempt in range(1, max_retries + 1):
        try:
            response = GEMINI_CLIENT.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)
        except json.JSONDecodeError:
            pass
        except Exception:
            pass
        if attempt < max_retries:
            time.sleep(retry_wait)
    return None


def call_gemini_json_cached(prompt: str) -> Any:
    key = f"gemini:v1:{MODEL}:{prompt}"
    return cached(key, lambda: call_gemini_json(prompt), format="json")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate search queries
# ─────────────────────────────────────────────────────────────────────────────

QUERY_PROMPT = """
You are a specialist researcher in the global beef meatpacking industry.

Generate EXACTLY 4 search queries about SLAUGHTERHOUSES operated
by {company} in {country}. A slaughterhouse = facility where LIVE ANIMALS ARE KILLED.

Query 1 and 2 - Target the OFFICIAL COMPANY WEBSITE or annual report:
  Write a SIMPLE and BROAD query - do NOT use site: operator, do NOT use | operator.
  The query should be 4-8 words maximum.
  Target: company website locations page, annual report, sustainability report,
  SEC/EDGAR filing if listed.

Query 3 - Target a GOVERNMENT AGRICULTURAL REGISTER:
  - Mexico    -> SENASICA directorio TIF rastros
  - USA       -> USDA FSIS slaughter establishment list
  - Australia -> DAFF export registered abattoir
  - France    -> DGAL abattoirs agrees CE
  - Japan     -> NLBC slaughterhouse list
  - Other     -> [country] official slaughterhouse register government

Query 4 - Target subsidiaries of the company:
  Example: "{company} subsidiaries slaughterhouses locations"

Return ONLY a valid JSON array of 4 strings.
"""


def generate_queries(company: str, country: str) -> list[str]:
    prompt = QUERY_PROMPT.replace("{company}", company).replace("{country}", country)
    result = call_gemini_json_cached(prompt)
    if not isinstance(result, list):
        return []
    return [str(q) for q in result if q]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — SerpAPI search
# ─────────────────────────────────────────────────────────────────────────────

def serpapi_search(
    queries: list[str],
    gl: str = "us",
    hl: str = "en",
    location: str = "United States",
    results_per_query: int = 4,
) -> list[dict]:
    """Run SerpAPI on each query. Deduplicates URLs and excludes social media."""
    all_results: list[dict] = []
    seen: set[str] = set()

    for query in queries:
        try:
            resp = requests.get(
                "https://serpapi.com/search",
                params={
                    "q": query,
                    "api_key": SERPAPI_API_KEY,
                    "num": results_per_query,
                    "gl": gl,
                    "hl": hl,
                    "location": location,
                    "filter": "0",
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("organic_results", [])[:results_per_query]
        except requests.RequestException:
            results = []

        for r in results:
            url = r.get("link", "")
            if url and url not in seen and not any(d in url for d in EXCLUDED_DOMAINS):
                seen.add(url)
                all_results.append({
                    "query": query,
                    "title": r.get("title", ""),
                    "source_urls": [url],
                    "snippet": r.get("snippet", ""),
                    "score": 7.0,
                    "type": "auto",
                })
        time.sleep(1)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Link scoring
# ─────────────────────────────────────────────────────────────────────────────

SCORING_PROMPT = """
You receive a list of URLs about slaughterhouses operated by {company}.
Select the MOST RELEVANT links (max 15) for finding slaughterhouse data.

SCORING (0-10):
  10 : Official government register (SENASICA, FSIS, DAFF, DGAL...)
  8-9: SEC/annual report listing plants with addresses
  7-8: Company website with plant/locations page
  5-6: Trade press with specific plant data
  3-4: General mention
  0  : Irrelevant (retail, office, social media)

Keep at least: 1 government source, 1 company source, 1 media source.

For each selected link return an object with:
  - url, title, score (0-10)
  - source_type: 'government'|'company'|'media'|'other'
  - expected_format: 'PDF'|'HTML'|'CSV'

Return ONLY a valid JSON array.
"""


def score_links(all_results: list[dict], company: str, country: str) -> list[dict]:
    """Score and filter the URLs returned by SerpAPI."""
    links_input = [
        {"title": r["title"][:80], "url": r["source_urls"][0], "snippet": r["snippet"][:100]}
        for r in all_results
    ]
    prompt = (
        SCORING_PROMPT.replace("{company}", company)
        + f"\n\nCompany: {company} | Country: {country}\n"
        + f"Links to evaluate:\n{json.dumps(links_input, ensure_ascii=False)}"
    )
    scored = call_gemini_json_cached(prompt)
    if not isinstance(scored, list):
        # fall back to raw links
        scored = [
            {"url": r["source_urls"][0], "title": r["title"],
             "score": r["score"], "source_type": "other"}
            for r in all_results
        ]

    return [
        {
            "source_urls": [l["url"]],
            "title": l.get("title", ""),
            "type": l.get("source_type", "other"),
            "score": l.get("score", 5.0),
            "local_path": None,
        }
        for l in scored
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Pre-flight HTTP check (detects 403/404/timeout BEFORE extraction)
# ─────────────────────────────────────────────────────────────────────────────

def _headers_for(url: str) -> dict:
    """Domain-aware headers (SEC requires a contact email)."""
    if "sec.gov" in url.lower():
        return {
            "User-Agent": "OSFL Research osfl@example.org",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }
    return HEADERS


def preflight_check(links: list[dict]) -> list[dict]:
    """Return the subset of links that respond with HTTP error or fail to reach."""
    problematic = []
    for i, link in enumerate(links):
        url = link["source_urls"][0]
        try:
            r = requests.head(url, headers=_headers_for(url), timeout=10, allow_redirects=True)
            status: Any = r.status_code
        except requests.RequestException as e:
            status = f"ERR ({type(e).__name__})"

        ok = isinstance(status, int) and status < 400
        if not ok:
            problematic.append({
                "index": i, "status": status, "url": url, "title": link["title"],
            })
        time.sleep(0.3)
    return problematic


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Format detection & content extraction
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(url: str) -> str:
    u = url.lower()
    if u.endswith(".pdf") or "/pdf/" in u:
        return "PDF"
    if u.endswith(".csv"):
        return "CSV"
    if u.endswith((".xlsx", ".xls")):
        return "EXCEL"
    try:
        head = requests.head(url, headers=_headers_for(url), timeout=10, allow_redirects=True)
        ct = head.headers.get("Content-Type", "").lower()
        if "pdf" in ct:
            return "PDF"
        if "csv" in ct:
            return "CSV"
        if "excel" in ct or "spreadsheet" in ct:
            return "EXCEL"
        if "html" in ct:
            return "HTML"
    except requests.RequestException:
        pass
    return "HTML"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    lines, chunks, buf = text.split("\n"), [], ""
    for line in lines:
        if len(buf) + len(line) + 1 > chunk_size and buf:
            chunks.append(buf)
            buf = buf[-overlap:] + "\n" + line
        else:
            buf = (buf + "\n" + line) if buf else line
    if buf.strip():
        chunks.append(buf)
    return chunks


def _extract_html(url: str, max_chars: int = 6000) -> list[str]:
    from bs4 import BeautifulSoup
    try:
        r = requests.get(url, headers=_headers_for(url), timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        lines = [l for l in soup.get_text("\n", strip=True).split("\n") if l.strip()]
        return _chunk_text("\n".join(lines)[:max_chars])
    except Exception:
        return []


def _tables_to_text(page) -> str:
    out = []
    try:
        for tbl in page.find_tables().tables:
            for row in tbl.extract():
                cells = [(c or "").strip().replace("\n", " ") for c in row]
                out.append(" | ".join(cells))
    except Exception:
        return ""
    return "\n".join(out)


def _extract_pdf_bytes(content: bytes, max_pages: int = 25, max_chars: int = 12000) -> list[str]:
    import fitz
    from PIL import Image
    import pytesseract
    try:
        pdf = fitz.open(stream=content, filetype="pdf")
    except Exception:
        return []
    n = min(len(pdf), max_pages)

    # 1) Tables
    parts = []
    for i in range(n):
        t = _tables_to_text(pdf[i])
        if t.strip():
            parts.append(f"--- Page {i+1} (table) ---\n{t}")
    text = "\n".join(parts)

    # 2) Plain text fallback
    if len(text.strip()) < 100:
        parts = []
        for i in range(n):
            pt = pdf[i].get_text()
            if pt.strip():
                parts.append(f"--- Page {i+1} ---\n{pt}")
        text = "\n".join(parts)

    # 3) OCR last-resort
    if len(text.strip()) < 100:
        parts = []
        for i in range(n):
            pix = pdf[i].get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            parts.append(f"--- Page {i+1} (OCR) ---\n"
                         + pytesseract.image_to_string(img, lang="eng+spa"))
        text = "\n".join(parts)

    pdf.close()
    return _chunk_text(text[:max_chars])


def _extract_pdf(url: str) -> list[str]:
    try:
        r = requests.get(url, headers=_headers_for(url), timeout=30)
        if r.status_code == 403:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        return _extract_pdf_bytes(r.content)
    except Exception:
        return []


def _extract_pdf_local(filepath: str) -> list[str]:
    try:
        with open(filepath, "rb") as f:
            return _extract_pdf_bytes(f.read(), max_chars=15000)
    except Exception:
        return []


def _extract_pdf_from_bytes(content: bytes) -> list[str]:
    """For uploaded files (Streamlit returns bytes, not a path)."""
    return _extract_pdf_bytes(content, max_chars=15000)


def _df_to_chunks(df) -> list[str]:
    text = f"Columns: {list(df.columns)}\n\n{df.to_string(index=False)}"
    return _chunk_text(text[:10000])


def _extract_csv(url: str, max_rows: int = 200) -> list[str]:
    import pandas as pd
    try:
        r = requests.get(url, headers=_headers_for(url), timeout=30)
        r.raise_for_status()
        df = None
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(r.content), encoding=enc,
                                 nrows=max_rows, on_bad_lines="skip")
                break
            except Exception:
                continue
        if df is None:
            return []
        return _df_to_chunks(df)
    except Exception:
        return []


def _extract_excel(url: str, max_rows: int = 200) -> list[str]:
    import pandas as pd
    try:
        r = requests.get(url, headers=_headers_for(url), timeout=30)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), nrows=max_rows)
        return _df_to_chunks(df)
    except Exception:
        return []


def extract_content(url: str, local_path: Optional[str] = None,
                    local_bytes: Optional[bytes] = None) -> tuple:
    """Return (chunks, format, total_chars).

    If local_bytes is provided (e.g. from a Streamlit uploader), use it as a PDF.
    If local_path is provided, read the file as a PDF.
    Otherwise, fetch the URL.
    """
    if local_bytes is not None:
        chunks = _extract_pdf_from_bytes(local_bytes)
        return chunks, "PDF", sum(len(c) for c in chunks)
    if local_path:
        chunks = _extract_pdf_local(local_path)
        return chunks, "PDF", sum(len(c) for c in chunks)

    fmt = detect_format(url)
    if fmt == "PDF":
        chunks = _extract_pdf(url)
    elif fmt == "CSV":
        chunks = _extract_csv(url)
    elif fmt == "EXCEL":
        chunks = _extract_excel(url)
    else:
        chunks = _extract_html(url)
        if not chunks:
            chunks = _extract_pdf(url)
            fmt = "PDF"
    return chunks, fmt, sum(len(c) for c in chunks)


def extract_content_cached(url: str, local_path: Optional[str] = None) -> tuple:
    """Cached variant. Note: local_bytes can't be cached easily — bypass cache for uploads."""
    key = f"extract:v1:{url}:{local_path}"
    return cached(key, lambda: extract_content(url, local_path=local_path), format="pickle")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Structured extraction (slaughterhouses from a source's text)
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
You are a specialist in global beef slaughter industry data extraction.

Company: {company} | Source format: {fmt}

TASK: Extract slaughterhouses DIRECTLY OPERATED or OWNED by {company}
or one of its subsidiaries.

SEARCH FOR:
- TIF numbers (Mexico), FSIS Est. numbers (USA), DAFF Est. (Australia)
- Tables listing facilities with addresses or cities
- Mentions of "slaughter", "rastro", "sacrificio", "abattoir", "harvest facility"

EXCLUDE STRICTLY:
- Pure processing/transformation factories (no live animal slaughter)
- Offices, laboratories, cold storage, distribution centers, retail butcheries

================  CAPACITY vs THROUGHPUT  ================
CAPACITY = the MAXIMUM theoretical processing potential of the facility.
  - Fixed characteristic of the plant.
  - Usually expressed per day or per hour.
  - Signals: "capacity", "capacidad", "maximum", "installed", "design capacity".

THROUGHPUT = the ACTUAL number of animals slaughtered over a given period.
  - Varies year to year.
  - Usually expressed per year.
  - A SERIES of different values for consecutive years is ALWAYS throughput.

DECISION RULE: if unsure, classify as throughput and set
classification_uncertain = true. Never default an ambiguous figure to capacity.
==========================================================

Return ONLY valid JSON:
{{
  "slaughterhouses": [
    {{
      "facility_name": "string or null",
      "operator": "string or null",
      "address": "string or null",
      "city": "string or null",
      "country": "string or null",
      "species": ["cattle"],
      "capacity": {{ "value": 0, "unit": "head/day", "year_reported": null }},
      "throughput": [
        {{ "value": 0, "unit": "head/year", "year": 2023 }}
      ],
      "classification_uncertain": false,
      "establishment_number": "string or null",
      "operational_status": "active",
      "export_certified": false,
      "confidence_score": 0.0,
      "confidence_reason": "string"
    }}
  ],
  "excluded": [
    {{ "facility_name": "string",
       "reason": "processing only / office / cold storage / laboratory / retail / other" }}
  ],
  "source_quality": "high|medium|low"
}}
"""


def _extract_abattoirs_from_chunk(chunk: str, company: str, fmt: str) -> dict:
    empty = {"slaughterhouses": [], "excluded": [], "source_quality": "low"}
    if not chunk or len(chunk.strip()) < 50:
        return empty
    prompt = (
        EXTRACTION_PROMPT.replace("{company}", company).replace("{fmt}", fmt)
        + f"\n\nSOURCE TEXT:\n{chunk[:MAX_CHARS_PER_CHUNK]}"
    )
    result = call_gemini_json_cached(prompt)
    return result if isinstance(result, dict) else empty


def extract_abattoirs_from_source(item: dict, company: str,
                                  sleep_between_chunks: int = 5) -> dict:
    """Run Gemini extraction over all chunks of a source; dedupe within source."""
    chunks = item.get("chunks", [])
    all_sh, all_excl, seen_local = [], [], set()
    for i, chunk in enumerate(chunks, 1):
        result = _extract_abattoirs_from_chunk(chunk, company, item["format"])
        for s in result.get("slaughterhouses", []):
            key = (s.get("establishment_number") or s.get("facility_name") or "").lower().strip()
            if key and key not in seen_local:
                seen_local.add(key)
                all_sh.append(s)
        all_excl.extend(result.get("excluded", []))
        if i < len(chunks):
            time.sleep(sleep_between_chunks)
    return {
        "slaughterhouses": all_sh,
        "excluded": all_excl,
        "source_quality": "high" if all_sh else "low",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Cross-source synthesis (Gemini deduplication & enrichment)
# ─────────────────────────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """
You are a data analyst specialising in the global beef slaughter industry.

You have slaughterhouse data extracted from MULTIPLE sources about {company}.
Produce a FINAL DEDUPLICATED and ENRICHED list.

RULES:
1. MERGE duplicates - same facility across sources
   (match priority: establishment_number > city+country > facility_name similarity)
2. ENRICH - combine complementary fields. CAPACITY and THROUGHPUT are SEPARATE.
3. THROUGHPUT is a list - merge entries from all sources, one per year.
4. INCREASE confidence_score if a facility appears in 2+ sources.
5. FLAG conflicts.
6. Keep ONLY slaughterhouses - discard processing-only facilities.

Return ONLY valid JSON:
{{ "final_slaughterhouses": [...], "synthesis_notes": "brief summary" }}
"""


def synthesize(all_abattoirs: list[dict], all_sources: list[dict], company: str) -> list[dict]:
    if not all_abattoirs:
        return []

    synthesis_input = json.dumps({
        "company": company,
        "sources_used": [
            {"urls": s["source_urls"], "type": s["type"], "format": s["format"]}
            for s in all_sources
        ],
        "raw_abattoirs": all_abattoirs,
    }, ensure_ascii=False, indent=2)

    prompt = (
        SYNTHESIS_PROMPT.replace("{company}", company)
        + f"\n\nCompany: {company}\nRaw extracted data:\n{synthesis_input}\n\n"
        + "Produce the final deduplicated, enriched slaughterhouse list."
    )

    result = call_gemini_json_cached(prompt)
    if not isinstance(result, dict):
        return all_abattoirs
    return result.get("final_slaughterhouses", [])


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Geocoding (Google Places primary, Nominatim fallback)
# ─────────────────────────────────────────────────────────────────────────────

COUNTRY_TO_ISO = {
    "france": "FR", "united states": "US", "usa": "US", "us": "US",
    "australia": "AU", "mexico": "MX", "mexique": "MX",
    "germany": "DE", "allemagne": "DE", "deutschland": "DE",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB",
    "spain": "ES", "italy": "IT", "netherlands": "NL",
    "belgium": "BE", "ireland": "IE", "canada": "CA",
    "brazil": "BR", "argentina": "AR", "china": "CN",
    "japan": "JP", "denmark": "DK", "poland": "PL", "sweden": "SE",
}


def _google_places_search(query: str, country_code: Optional[str] = None) -> dict:
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.types,places.id",
    }
    body: dict = {"textQuery": query, "maxResultCount": 5}
    if country_code:
        body["regionCode"] = country_code.lower()
    r = requests.post(url, headers=headers, json=body, timeout=15)
    r.raise_for_status()
    return r.json()


def _google_places_search_cached(query: str, country_code: Optional[str]) -> dict:
    key = f"google_places:v1:{query}:{country_code}"
    return cached(key, lambda: _google_places_search(query, country_code), format="json")


def _normalize(s: Optional[str]) -> str:
    import unicodedata
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return s.lower().strip()


def geocode_google_first(facility: dict, default_company: str = "") -> None:
    """Geocode a facility in-place. Sets latitude/longitude/coords_source."""
    name = facility.get("facility_name") or ""
    city = facility.get("city") or ""
    country = facility.get("country") or ""
    operator = facility.get("operator") or default_company

    iso = COUNTRY_TO_ISO.get(_normalize(country))

    query_parts = []
    seen = set()
    for p in [operator, name, city, "meat"]:
        if p and p.lower() not in seen:
            query_parts.append(p)
            seen.add(p.lower())
    query = " ".join(query_parts)

    # Step 1: Google Places
    try:
        data = _google_places_search_cached(query, country_code=iso)
        places = data.get("places", [])
        if places:
            best = places[0]
            loc = best.get("location") or {}
            g_lat, g_lon = loc.get("latitude"), loc.get("longitude")
            if g_lat and g_lon:
                facility["latitude"] = g_lat
                facility["longitude"] = g_lon
                facility["geocoding_quality"] = "google_places"
                facility["coords_source"] = "google_places"
                facility["google_address"] = best.get("formattedAddress", "")
                facility["google_name"] = (best.get("displayName") or {}).get("text", "")
                facility["google_types"] = ", ".join(best.get("types", []))
                return
    except Exception:
        pass

    # Step 2: Nominatim fallback
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
        geolocator = Nominatim(user_agent="osfl_slaughterhouse_tool")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)
        cc = (iso or "").lower() or None
        for q in [f"{name}, {city}, {country}", f"{city}, {country}"]:
            if "None" in q or "null" in q:
                continue
            result = geocode(q, country_codes=cc) if cc else geocode(q)
            if result:
                facility["latitude"] = result.latitude
                facility["longitude"] = result.longitude
                facility["geocoding_quality"] = "nominatim"
                facility["coords_source"] = "nominatim"
                return
    except Exception:
        pass

    facility["geocoding_quality"] = "failed"
    facility["coords_source"] = None

def flag_duplicate_addresses(facilities: list[dict], coord_tolerance_m: int = 100) -> int:
    from geopy.distance import geodesic

    # Reset previous flags
    for f in facilities:
        f["duplicate_address_flag"] = False
        f["duplicate_address_group"] = None
        f["duplicate_with"] = []

    # Only consider geocoded facilities
    geo = [f for f in facilities if f.get("latitude") and f.get("longitude")]

    # Group by proximity / shared address
    groups: list[list[dict]] = []

    def _norm_addr(f: dict) -> str:
        return _normalize(f.get("google_address") or f.get("address") or "")

    for f in geo:
        placed = False
        f_addr = _norm_addr(f)
        for group in groups:
            for g in group:
                same_addr = bool(f_addr) and f_addr == _norm_addr(g)
                try:
                    dist = geodesic(
                        (f["latitude"], f["longitude"]),
                        (g["latitude"], g["longitude"]),
                    ).meters
                except Exception:
                    dist = float("inf")
                if same_addr or dist <= coord_tolerance_m:
                    group.append(f)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([f])

    # Flag groups with more than one member
    n_flagged = 0
    group_id = 0
    for group in groups:
        if len(group) > 1:
            names = [g.get("facility_name", "?") for g in group]
            for g in group:
                g["duplicate_address_flag"] = True
                g["duplicate_address_group"] = group_id
                g["duplicate_with"] = [n for n in names if n != g.get("facility_name", "?")]
                n_flagged += 1
            group_id += 1

    return n_flagged


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Capacity refinement (targeted search per facility)
# ─────────────────────────────────────────────────────────────────────────────

CAPACITY_REFINEMENT_PROMPT = """
You are extracting CAPACITY data for ONE SPECIFIC slaughterhouse.

Target facility: {facility_name}
Operator: {operator}
Location: {city}, {country}

================  CRITICAL: FACILITY vs GROUP  ================
The source text often mentions BOTH the parent company's TOTAL figures
AND the specific facility's figures. You must extract ONLY the
SPECIFIC FACILITY's capacity, NEVER the parent company's total.

Signals that a figure refers to the GROUP/TOTAL (DO NOT extract):
- "the group has a total capacity of X"
- "globally", "worldwide", "company-wide"
- Figures mentioned alongside company financials

Signals that a figure refers to THIS FACILITY (extract):
- "the {facility_name} plant has a capacity of X"
- "the {city} facility processes X per day"
- A figure given immediately next to the facility name or city
================================================================

Return ONLY valid JSON:
{{
  "found": true|false,
  "capacity": {{ "value": integer, "unit": "head/day|head/hour|head/week|head/year",
                 "year_reported": integer or null,
                 "species": "cattle|pigs|sheep|poultry|mixed|null" }},
  "evidence_quote": "short quote from the source (max 30 words)",
  "scope_check": "facility | group | uncertain",
  "confidence": 0.0-1.0
}}
If no facility-specific capacity found, return: {{ "found": false }}
"""


def refine_capacity(facility: dict, gl: str, hl: str, location: str,
                    default_company: str = "") -> dict:
    """Run a targeted SerpAPI + Gemini search for capacity. Returns a log dict."""
    name = facility.get("facility_name") or ""
    city = facility.get("city") or ""
    country = facility.get("country") or ""
    operator = facility.get("operator") or default_company

    log = {"queries": [], "candidate_urls": [], "findings": [], "outcome": "no_change"}

    queries = []
    if operator and city:
        queries.append(f"{operator} {city} capacity slaughterhouse")
    if name and city and name.lower() != operator.lower():
        queries.append(f"{operator} {city} capacity head per day")
    if not queries:
        log["outcome"] = "insufficient_info"
        return log
    log["queries"] = queries

    # SerpAPI
    candidate_urls, seen = [], set()
    for q in queries:
        try:
            resp = requests.get("https://serpapi.com/search", params={
                "q": q, "api_key": SERPAPI_API_KEY, "num": 3,
                "gl": gl, "hl": hl, "location": location, "filter": "0",
            }, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("organic_results", [])[:3]
        except requests.RequestException:
            continue
        for r in results:
            url = r.get("link", "")
            if url and url not in seen and not any(d in url for d in EXCLUDED_DOMAINS):
                seen.add(url)
                candidate_urls.append({"url": url, "title": r.get("title", "")})
        time.sleep(1)
    log["candidate_urls"] = candidate_urls

    if not candidate_urls:
        log["outcome"] = "no_candidates"
        return log

    findings = []
    refinement_prompt = CAPACITY_REFINEMENT_PROMPT.format(
        facility_name=name, operator=operator, city=city, country=country)

    for c in candidate_urls:
        chunks, fmt, n_chars = extract_content_cached(c["url"])
        if not chunks:
            continue
        for chunk in chunks[:3]:
            result = call_gemini_json_cached(
                refinement_prompt + f"\n\nSOURCE TEXT:\n{chunk[:10000]}"
            )
            if isinstance(result, dict) and result.get("found"):
                cap = result.get("capacity", {})
                if cap.get("value"):
                    findings.append({
                        "value": cap.get("value"),
                        "unit": cap.get("unit"),
                        "year": cap.get("year_reported"),
                        "evidence": result.get("evidence_quote", ""),
                        "confidence": result.get("confidence", 0),
                        "scope": result.get("scope_check", "uncertain"),
                        "source": c["url"],
                    })
                    break
        time.sleep(2)
    log["findings"] = findings

    existing = facility.get("capacity") or {}
    if not findings:
        log["outcome"] = "no_capacity_found"
    elif not existing.get("value"):
        best = max(findings, key=lambda f: f["confidence"])
        facility["capacity"] = {
            "value": best["value"],
            "unit": best["unit"],
            "year_reported": best["year"],
        }
        facility["capacity_evidence"] = best["evidence"]
        facility["capacity_source"] = best["source"]
        facility["capacity_added_by"] = "refinement"
        log["outcome"] = "capacity_filled"
    else:
        facility["capacity_alternatives"] = findings
        facility["capacity_review_needed"] = True
        log["outcome"] = "alternatives_recorded"
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Export to dataframe
# ─────────────────────────────────────────────────────────────────────────────

def build_export_dataframe(final_abattoirs: list[dict], company: str):
    """Flatten the final list to a tabular DataFrame (one row per throughput entry)."""
    import pandas as pd

    rows = []
    for a in sorted(final_abattoirs, key=lambda x: x.get("confidence_score", 0), reverse=True):
        cap = a.get("capacity") or {}
        throughputs = a.get("throughput") or [{}]
        base = {
            "Company": a.get("operator") or company,
            "Facility_name": a.get("facility_name", ""),
            "Est.#": a.get("establishment_number", ""),
            "Address": a.get("google_address"),
            "City": a.get("city", ""),
            "Country": a.get("country", ""),
            "Latitude": a.get("latitude", ""),
            "Longitude": a.get("longitude", ""),
            "Coords_source": a.get("coords_source", ""),
            "Google_name": a.get("google_name", ""),
            "Google_address": a.get("google_address", ""),
            "Species": ", ".join(a.get("species", [])),
            "Capacity_value": cap.get("value", ""),
            "Capacity_unit": cap.get("unit", ""),
            "Capacity_year": cap.get("year_reported", ""),
            "Capacity_evidence": a.get("capacity_evidence", ""),
            "Capacity_source": a.get("capacity_source", ""),
            "Status": a.get("operational_status", ""),
            "Export_certified": a.get("export_certified", ""),
            "Classification_uncertain": a.get("classification_uncertain", ""),
            "Confidence": a.get("confidence_score", ""),
            "Duplicate_address_flag": a.get("duplicate_address_flag", False),
            "Duplicate_with": " | ".join(a.get("duplicate_with", [])),
            "N_sources": a.get("n_sources", len(a.get("source_urls", []))),
            "Source_URLs": " | ".join(a.get("source_urls", [])),
        }
        for t in throughputs:
            rows.append({**base,
                         "Throughput_value": t.get("value", ""),
                         "Throughput_unit": t.get("unit", ""),
                         "Throughput_year": t.get("year", "")})
    return pd.DataFrame(rows)
