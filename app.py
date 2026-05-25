"""
Streamlit app — Slaughterhouse data extraction tool.

Oxford Sustainable Finance Lab.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Slaughterhouse Tool — OSFL",
    page_icon="🏭",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "configured": False,
        "company": "",
        "country": "",
        "gl": "us",
        "hl": "en",
        "location": "",
        "queries": [],
        "search_results": [],
        "scored_links": [],
        "problematic": [],
        "local_files": {},          # idx -> filepath (saved upload)
        "extracted_sources": [],    # list of {source_urls, title, format, chars, chunks, ...}
        "raw_abattoirs": [],
        "final_abattoirs": [],
        "exclusions": [],
        "refinement_logs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — API keys & cache
# ─────────────────────────────────────────────────────────────────────────────

def _get_secret(name: str) -> str:
    """Look up a secret in st.secrets first, then in env vars. Empty string if absent.

    Works in three deployment modes:
    1. Streamlit Cloud → uses st.secrets (set in dashboard).
    2. Local dev with .streamlit/secrets.toml → uses st.secrets.
    3. Local dev with env vars exported → uses os.environ.
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except (FileNotFoundError, AttributeError, KeyError):
        # st.secrets raises if no secrets.toml is present in local dev
        pass
    return os.environ.get(name, "")


# Auto-configure on first run if all keys are present in secrets/env.
# This avoids forcing users on Streamlit Cloud to click "Save configuration".
if not st.session_state.configured:
    _g = _get_secret("GEMINI_API_KEY")
    _s = _get_secret("SERPAPI_API_KEY")
    _p = _get_secret("GOOGLE_PLACES_API_KEY")
    if _g and _s and _p:
        try:
            pipeline.configure(
                cache_dir=_get_secret("CACHE_DIR") or str(Path.home() / ".osfl_slaughterhouse_cache"),
                gemini_api_key=_g,
                serpapi_key=_s,
                google_places_key=_p,
            )
            st.session_state.configured = True
        except Exception as e:
            st.warning(f"Auto-configuration failed: {e}. Please configure manually in the sidebar.")


with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    if st.session_state.configured:
        # Production mode: secrets loaded successfully, hide all API inputs.
        st.success("✓ API keys loaded from secrets.")
    else:
        # Fallback mode: no secrets found (local dev without .streamlit/secrets.toml).
        # Show inputs so the user can still develop locally.
        st.warning(
            "API keys not found in secrets. For Streamlit Cloud, add them in "
            "**Settings → Secrets**. For local dev, create `.streamlit/secrets.toml`."
        )
        gemini_key = st.text_input(
            "Gemini API key", type="password",
            value=_get_secret("GEMINI_API_KEY"),
            help="https://aistudio.google.com/apikey",
        )
        serpapi_key = st.text_input(
            "SerpAPI key", type="password",
            value=_get_secret("SERPAPI_API_KEY"),
            help="https://serpapi.com/manage-api-key",
        )
        google_key = st.text_input(
            "Google Places API key", type="password",
            value=_get_secret("GOOGLE_PLACES_API_KEY"),
            help="https://console.cloud.google.com — Places API (New)",
        )
        cache_dir = st.text_input(
            "Cache directory",
            value=_get_secret("CACHE_DIR") or str(Path.home() / ".osfl_slaughterhouse_cache"),
        )
        if st.button("Save configuration", type="primary", use_container_width=True):
            if not (gemini_key and serpapi_key and google_key):
                st.error("All three API keys are required.")
            else:
                pipeline.configure(
                    cache_dir=cache_dir,
                    gemini_api_key=gemini_key,
                    serpapi_key=serpapi_key,
                    google_places_key=google_key,
                )
                st.session_state.configured = True
                st.success("Configured.")
                st.rerun()

    if st.session_state.configured:
        st.divider()
        st.markdown("**Cache**")
        stats = pipeline.cache_stats()
        st.caption(f"{stats['files']} files · {stats['size_mb']} MB")
        if st.button("Clear cache", use_container_width=True):
            n = pipeline.cache_clear()
            st.success(f"Deleted {n} files.")

    st.divider()
    st.caption(
        "Need an overview of how the pipeline works? "
        "See the **About** section at the bottom of the page."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏭 Slaughterhouse Data Extraction Tool")
st.caption(
    "Oxford Sustainable Finance Lab — automated extraction of slaughterhouse "
    "data for any meatpacking group, with cross-source synthesis and geocoding."
)

if not st.session_state.configured:
    st.info(
        "👈 Configure your API keys in the sidebar to get started. "
        "You need a Gemini key, a SerpAPI key, and a Google Places API key."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Target selection
# ─────────────────────────────────────────────────────────────────────────────

st.header("1. Target company")

c1, c2 = st.columns(2)
with c1:
    st.session_state.company = st.text_input(
        "Company name", value=st.session_state.company,
        placeholder="e.g. Bigard, SuKarne, Kilcoy, Danish Crown",
    )
    st.session_state.country = st.text_input(
        "Country", value=st.session_state.country,
        placeholder="e.g. France, Mexico, Australia, Denmark",
    )

with c2:
    GL_OPTIONS = {
        "fr": "France", "us": "United States", "mx": "Mexico", "au": "Australia",
        "de": "Germany", "uk": "United Kingdom", "es": "Spain", "it": "Italy",
        "nl": "Netherlands", "be": "Belgium", "ie": "Ireland", "ca": "Canada",
        "br": "Brazil", "ar": "Argentina", "jp": "Japan", "dk": "Denmark",
        "pl": "Poland", "se": "Sweden",
    }
    HL_OPTIONS = {
        "en": "English", "fr": "French", "es": "Spanish", "de": "German",
        "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "nl": "Dutch",
        "da": "Danish", "sv": "Swedish", "pl": "Polish",
    }
    st.session_state.gl = st.selectbox(
        "Google country code (gl)",
        list(GL_OPTIONS.keys()),
        format_func=lambda c: f"{c} — {GL_OPTIONS[c]}",
        index=list(GL_OPTIONS.keys()).index(st.session_state.gl)
            if st.session_state.gl in GL_OPTIONS else 0,
    )
    st.session_state.hl = st.selectbox(
        "Results language (hl)",
        list(HL_OPTIONS.keys()),
        format_func=lambda c: f"{c} — {HL_OPTIONS[c]}",
        index=list(HL_OPTIONS.keys()).index(st.session_state.hl)
            if st.session_state.hl in HL_OPTIONS else 0,
    )
    st.session_state.location = st.text_input(
        "SerpAPI location", value=st.session_state.location or st.session_state.country,
        placeholder="e.g. France, United States, Mexico",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Generate queries & search
# ─────────────────────────────────────────────────────────────────────────────

st.header("2. Web search")

can_search = bool(st.session_state.company and st.session_state.country)

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🔍 Generate queries & search", disabled=not can_search, type="primary"):
        with st.spinner("Generating queries with Gemini..."):
            st.session_state.queries = pipeline.generate_queries(
                st.session_state.company, st.session_state.country,
            )
        if not st.session_state.queries:
            st.error("Failed to generate queries — check Gemini key & quota.")
        else:
            with st.spinner(f"Running {len(st.session_state.queries)} SerpAPI searches..."):
                st.session_state.search_results = pipeline.serpapi_search(
                    st.session_state.queries,
                    gl=st.session_state.gl,
                    hl=st.session_state.hl,
                    location=st.session_state.location or st.session_state.country,
                )

with c2:
    if st.session_state.queries:
        st.markdown("**Generated queries**")
        for i, q in enumerate(st.session_state.queries, 1):
            st.markdown(f"{i}. `{q}`")
    if st.session_state.search_results:
        st.markdown(
            f"**{len(st.session_state.search_results)} unique links** from SerpAPI"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Score links & pre-flight check
# ─────────────────────────────────────────────────────────────────────────────

st.header("3. Score & check links")

if not st.session_state.search_results:
    st.caption("Run step 2 first.")
else:
    if st.button("🏆 Score links with Gemini"):
        with st.spinner("Scoring links..."):
            st.session_state.scored_links = pipeline.score_links(
                st.session_state.search_results,
                st.session_state.company,
                st.session_state.country,
            )
        with st.spinner(f"Pre-flight HTTP check on {len(st.session_state.scored_links)} links..."):
            st.session_state.problematic = pipeline.preflight_check(st.session_state.scored_links)
        # reset uploads from previous run
        st.session_state.local_files = {}

    if st.session_state.scored_links:
        st.markdown(f"**{len(st.session_state.scored_links)} links scored**")
        problematic_idx = {p["index"] for p in st.session_state.problematic}

        for i, link in enumerate(st.session_state.scored_links):
            icon = {"government": "🏛️", "company": "🏢", "media": "📰"}.get(link["type"], "🔗")
            status_badge = "❌" if i in problematic_idx else "✅"
            url = link["source_urls"][0]
            score = link.get("score", 0)
            st.markdown(
                f"{status_badge} {icon} **[{score}/10]** {link['title'][:70]}  \n"
                f"<span style='color:#888'>{url}</span>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Resolve problematic links (manual upload)
# ─────────────────────────────────────────────────────────────────────────────

st.header("4. Resolve unreachable links")

if not st.session_state.scored_links:
    st.caption("Run step 3 first.")
elif not st.session_state.problematic:
    st.success("All links responded OK — no manual action needed.")
else:
    st.warning(
        f"{len(st.session_state.problematic)} link(s) returned HTTP errors. "
        "You can either upload the file manually, or skip them."
    )

    for p in st.session_state.problematic:
        idx = p["index"]
        with st.expander(f"❌ [{idx}] HTTP {p['status']} — {p['title'][:60]}", expanded=False):
            st.markdown(f"**URL:** {p['url']}")
            st.markdown(
                "Download the file in your browser, then upload it here. "
                "The tool will use the local copy instead of the broken URL."
            )
            uploaded = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                key=f"upload_{idx}",
            )
            if uploaded is not None:
                # Save to a temp file so the pipeline can read it as a path
                tmp_dir = Path(tempfile.gettempdir()) / "osfl_uploads"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / f"link_{idx}_{uploaded.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
                st.session_state.local_files[idx] = str(tmp_path)
                # Update the scored_links entry so the extraction step picks it up
                st.session_state.scored_links[idx]["local_path"] = str(tmp_path)
                st.success(f"✓ Saved as {tmp_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Extract content & structured data
# ─────────────────────────────────────────────────────────────────────────────

st.header("5. Extract & structure data")

if not st.session_state.scored_links:
    st.caption("Run steps 2-3 first.")
else:
    # Let the user choose which links to extract
    # Default: skip problematic links that have no local upload
    problematic_idx = {p["index"] for p in st.session_state.problematic}
    default_skip = problematic_idx - set(st.session_state.local_files.keys())

    options = {}
    for i, link in enumerate(st.session_state.scored_links):
        skip = i in default_skip
        label = f"[{link.get('score',0)}/10] {link['title'][:60]}"
        if i in problematic_idx and i not in st.session_state.local_files:
            label += "  ⚠️ unreachable"
        elif i in st.session_state.local_files:
            label += "  📎 local upload"
        options[i] = (label, not skip)

    st.markdown("**Select links to process**")
    selected_indices = []
    for i, (label, default) in options.items():
        if st.checkbox(label, value=default, key=f"select_{i}"):
            selected_indices.append(i)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🚀 Run extraction", type="primary",
                     disabled=not selected_indices):
            extracted = []
            progress = st.progress(0.0, text="Starting extraction...")

            for n, i in enumerate(selected_indices, 1):
                link = st.session_state.scored_links[i]
                url = link["source_urls"][0]
                local_path = link.get("local_path")
                progress.progress(
                    (n - 1) / len(selected_indices),
                    text=f"[{n}/{len(selected_indices)}] {link['title'][:50]}",
                )
                try:
                    chunks, fmt, n_chars = pipeline.extract_content(
                        url, local_path=local_path,
                    )
                except Exception as e:
                    st.warning(f"Extraction failed for {url}: {e}")
                    continue
                if not chunks:
                    continue
                extracted.append({
                    "source_urls": [url],
                    "title": link["title"],
                    "type": link["type"],
                    "score": link["score"],
                    "format": fmt,
                    "chars": n_chars,
                    "chunks": chunks,
                })

            progress.progress(1.0, text="Content extracted.")
            st.session_state.extracted_sources = extracted

            # Now structured extraction with Gemini
            company = st.session_state.company
            all_abattoirs = []
            seen_global: dict[str, int] = {}

            prog2 = st.progress(0.0, text="Starting structured extraction...")
            for n, item in enumerate(extracted, 1):
                prog2.progress(
                    (n - 1) / max(len(extracted), 1),
                    text=f"[{n}/{len(extracted)}] {item['title'][:50]}",
                )
                result = pipeline.extract_abattoirs_from_source(item, company)
                item["exclusions"] = result.get("excluded", [])

                for a in result.get("slaughterhouses", []):
                    key = (a.get("establishment_number") or a.get("facility_name") or "").lower().strip()
                    if key in seen_global:
                        existing = all_abattoirs[seen_global[key]]
                        for u in item["source_urls"]:
                            if u not in existing.get("source_urls", []):
                                existing.setdefault("source_urls", []).append(u)
                    elif key:
                        a["source_urls"] = list(item["source_urls"])
                        a["source_format"] = item["format"]
                        seen_global[key] = len(all_abattoirs)
                        all_abattoirs.append(a)
                item["chunks"] = []     # free memory
            prog2.progress(1.0, text="Done.")

            st.session_state.raw_abattoirs = all_abattoirs

            # Cross-source synthesis
            with st.spinner("Synthesising across sources..."):
                final = pipeline.synthesize(all_abattoirs, extracted, company)
            st.session_state.final_abattoirs = final

            # Collect exclusions
            all_exclusions = []
            for item in extracted:
                src = item["source_urls"][0] if item.get("source_urls") else ""
                for excl in item.get("exclusions", []):
                    if isinstance(excl, dict):
                        all_exclusions.append({
                            "Excluded facility": excl.get("facility_name", "Unknown"),
                            "Reason": excl.get("reason", "n/a"),
                            "Source": src[:60],
                        })
            st.session_state.exclusions = all_exclusions

    with c2:
        if st.session_state.final_abattoirs:
            st.success(
                f"Extracted **{len(st.session_state.final_abattoirs)}** unique slaughterhouses "
                f"from **{len(st.session_state.extracted_sources)}** sources."
            )

    if st.session_state.final_abattoirs:
        st.dataframe(
            pd.DataFrame([
                {
                    "Facility": a.get("facility_name", "?"),
                    "City": a.get("city", ""),
                    "Country": a.get("country", ""),
                    "Est.#": a.get("establishment_number", ""),
                    "Capacity": (
                        f"{(a.get('capacity') or {}).get('value','')} "
                        f"{(a.get('capacity') or {}).get('unit','')}"
                        if (a.get("capacity") or {}).get("value")
                        else "—"
                    ),
                    "Confidence": f"{a.get('confidence_score', 0):.0%}",
                    "Status": a.get("operational_status", ""),
                }
                for a in st.session_state.final_abattoirs
            ]),
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Capacity refinement (targeted search)
# ─────────────────────────────────────────────────────────────────────────────

st.header("6. Refine capacity (optional)")

if not st.session_state.final_abattoirs:
    st.caption("Run step 5 first.")
else:
    source_data = st.session_state.final_abattoirs

    options_for_refine = {}
    for i, a in enumerate(source_data):
        cap = a.get("capacity") or {}
        cap_str = f"{cap.get('value','?')} {cap.get('unit','')}" if cap.get("value") else "no capacity"
        options_for_refine[i] = f"[{i}] {a.get('facility_name','?')} — {a.get('city','?')} ({cap_str})"

    refine_filter = st.radio(
        "Which facilities?",
        ["Missing capacity only", "All", "Custom selection"],
        horizontal=True,
    )

    if refine_filter == "Missing capacity only":
        targets = [i for i, a in enumerate(source_data)
                   if not (a.get("capacity") or {}).get("value")]
    elif refine_filter == "All":
        targets = list(range(len(source_data)))
    else:
        targets = st.multiselect(
            "Pick facilities", list(options_for_refine.keys()),
            format_func=lambda i: options_for_refine[i],
        )

    if targets:
        st.caption(f"{len(targets)} facility(ies) selected.")

    if st.button("🎯 Refine capacity", disabled=not targets):
        logs = []
        progress = st.progress(0.0, text="Starting refinement...")
        for n, idx in enumerate(targets, 1):
            a = source_data[idx]
            progress.progress(
                (n - 1) / len(targets),
                text=f"[{n}/{len(targets)}] {a.get('facility_name','?')}",
            )
            log = pipeline.refine_capacity(
                a,
                gl=st.session_state.gl,
                hl=st.session_state.hl,
                location=st.session_state.location or st.session_state.country,
                default_company=st.session_state.company,
            )
            log["index"] = idx
            log["facility_name"] = a.get("facility_name", "?")
            logs.append(log)
        progress.progress(1.0, text="Done.")
        st.session_state.refinement_logs = logs

    if st.session_state.refinement_logs:
        st.markdown("**Refinement results**")
        for log in st.session_state.refinement_logs:
            with st.expander(
                f"[{log['index']}] {log['facility_name']} — {log['outcome']}",
                expanded=False,
            ):
                if log["queries"]:
                    st.markdown("**Queries used:** " + ", ".join(f"`{q}`" for q in log["queries"]))
                if log["findings"]:
                    st.markdown(f"**{len(log['findings'])} finding(s):**")
                    for f in log["findings"]:
                        st.markdown(
                            f"- **{f['value']:,} {f['unit']}** (conf {f['confidence']:.0%}, "
                            f"scope={f['scope']})  \n"
                            f"  > {f['evidence']}  \n"
                            f"  [source]({f['source']})"
                        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Geocoding
# ─────────────────────────────────────────────────────────────────────────────

st.header("7. Geocoding")

if not st.session_state.final_abattoirs:
    st.caption("Run step 5 first.")
else:
    source_data = st.session_state.final_abattoirs
    already_geocoded = sum(1 for a in source_data if a.get("latitude"))
    st.caption(f"{already_geocoded}/{len(source_data)} already geocoded.")

    geocode_filter = st.radio(
        "Which facilities?",
        ["Not yet geocoded", "All", "Custom selection"],
        horizontal=True, key="geocode_filter",
    )

    if geocode_filter == "Not yet geocoded":
        targets = [i for i, a in enumerate(source_data) if not a.get("latitude")]
    elif geocode_filter == "All":
        targets = list(range(len(source_data)))
    else:
        targets = st.multiselect(
            "Pick facilities",
            list(range(len(source_data))),
            format_func=lambda i: f"[{i}] {source_data[i].get('facility_name','?')} — "
                                   f"{source_data[i].get('city','?')}",
            key="geocode_multiselect",
        )

    if st.button("📍 Geocode (Google Places + Nominatim fallback)", disabled=not targets):
        progress = st.progress(0.0, text="Starting geocoding...")
        for n, idx in enumerate(targets, 1):
            a = source_data[idx]
            progress.progress(
                (n - 1) / len(targets),
                text=f"[{n}/{len(targets)}] {a.get('facility_name','?')}",
            )
            pipeline.geocode_google_first(a, default_company=st.session_state.company)
        progress.progress(1.0, text="Done.")
        st.rerun()

    # Stats
    by_source = {}
    for a in source_data:
        s = a.get("coords_source") or "not_geocoded"
        by_source[s] = by_source.get(s, 0) + 1
    cols = st.columns(max(len(by_source), 1))
    for col, (src, n) in zip(cols, sorted(by_source.items())):
        col.metric(src, n)

    # Map
    geocoded = [a for a in source_data if a.get("latitude") and a.get("longitude")]
    if geocoded:
        st.subheader("Map")
        df_map = pd.DataFrame([
            {"lat": a["latitude"], "lon": a["longitude"],
             "name": a.get("facility_name", "?")}
            for a in geocoded
        ])
        st.map(df_map, latitude="lat", longitude="lon", size=80)


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Export
# ─────────────────────────────────────────────────────────────────────────────

st.header("8. Export")

if not st.session_state.final_abattoirs:
    st.caption("Run step 5 first.")
else:
    df = pipeline.build_export_dataframe(
        st.session_state.final_abattoirs, st.session_state.company,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    n_facilities = len(st.session_state.final_abattoirs)
    n_rows = len(df)
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Unique facilities", n_facilities)
    cB.metric("CSV rows", n_rows)
    cC.metric("Excluded", len(st.session_state.exclusions))
    cD.metric("Sources processed", len(st.session_state.extracted_sources))

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = f"slaughterhouses_{st.session_state.company}_{datetime.now():%Y%m%d}.csv"
    st.download_button(
        "💾 Download CSV", data=csv_bytes, file_name=filename, mime="text/csv",
        type="primary",
    )

    if st.session_state.exclusions:
        with st.expander(f"Excluded facilities ({len(st.session_state.exclusions)})"):
            st.dataframe(
                pd.DataFrame(st.session_state.exclusions),
                use_container_width=True, hide_index=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# About
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️ About this tool"):
    st.markdown("""
This tool extracts slaughterhouse data for a meatpacking group, using a pipeline of:

1. **Query generation** (Gemini) — builds focused search queries targeting company sites and government registers.
2. **Web search** (SerpAPI) — runs the queries and collects URLs.
3. **Link scoring** (Gemini) — ranks the URLs by usefulness.
4. **Pre-flight check** — detects unreachable links (403/404/timeout) so you can upload PDFs manually.
5. **Content extraction** — downloads HTML/PDF/CSV/Excel and chunks it into text.
6. **Structured extraction** (Gemini) — turns the text into a list of slaughterhouses with capacity/throughput.
7. **Cross-source synthesis** (Gemini) — deduplicates and enriches across sources.
8. **Capacity refinement** (Gemini + SerpAPI) — targeted search for missing capacities.
9. **Geocoding** — Google Places API as primary, Nominatim as fallback.
10. **Export** — flat CSV ready for analysis.

All Gemini and Google Places calls are cached on disk to make re-runs fast.

**Key conceptual distinction:** capacity = maximum theoretical processing potential (fixed per plant);
throughput = actual number slaughtered over a period (varies year to year). The pipeline keeps them separate.
""")
