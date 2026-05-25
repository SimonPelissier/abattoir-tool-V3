# Slaughterhouse Data Extraction Tool

Streamlit app for the Oxford Sustainable Finance Lab — extracts slaughterhouse data for any meatpacking group from public web sources, with cross-source synthesis and geocoding.

## Repository layout

```
.
├── app.py                    Streamlit UI (entry point)
├── pipeline.py               Pure-logic pipeline (no Streamlit calls)
├── requirements.txt          Python dependencies
├── packages.txt              System (apt) dependencies — needed for Tesseract OCR
├── secrets.toml.example      Template for API keys (DO NOT commit your filled version)
├── .gitignore
└── README.md
```

---

## Option A — Deploy on Streamlit Cloud (recommended)

This is the easiest setup for sharing the tool with colleagues at OSFL.

### 1. Push the repo to GitHub

```bash
git add app.py pipeline.py requirements.txt packages.txt .gitignore README.md secrets.toml.example
git commit -m "Initial deployment"
git push origin main
```

Make sure your `secrets.toml` (if you created one locally) is **never** committed — it's already in `.gitignore`.

### 2. Create the app on Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Select your repository, branch (`main`), and main file (`app.py`).
4. Before clicking **Deploy**, click **Advanced settings → Secrets**.
5. Paste your three API keys in TOML format:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   SERPAPI_API_KEY = "your-key-here"
   GOOGLE_PLACES_API_KEY = "your-key-here"
   ```
6. Click **Deploy**. The first build takes ~5 minutes (installs Python deps and apt packages).

### 3. Once deployed

- The app auto-loads API keys from secrets — users don't need to paste them.
- You get a public URL like `https://your-app.streamlit.app` to share.
- To update the app, just push to GitHub — Streamlit Cloud redeploys automatically.

### Important notes for Streamlit Cloud

- **Cache is ephemeral**: the disk cache is lost on every redeploy or auto-restart. The cache still helps within a session, but you'll re-pay API calls between sessions. For ~30 facilities per company that's small.
- **1 GB RAM limit** on the free tier: if you hit "Resource limits exceeded" during OCR on large PDFs, lower `max_pages` in `pipeline._extract_pdf`.
- **Public app**: anyone with the URL can use your API quotas. If that's a concern, you can make the app private from the dashboard.

---

## Option B — Run locally

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract (system dependency for OCR)

- macOS: `brew install tesseract tesseract-lang`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-fra`
- Windows: download from https://github.com/UB-Mannheim/tesseract/wiki

### 3. Provide your API keys

Two options:

**Option B.1 — `.streamlit/secrets.toml`** (mirrors Streamlit Cloud setup)
```bash
mkdir -p .streamlit
cp secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml with your keys
```

**Option B.2 — environment variables**
```bash
export GEMINI_API_KEY="..."
export SERPAPI_API_KEY="..."
export GOOGLE_PLACES_API_KEY="..."
```

### 4. Run

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

---

## How to get the API keys

| Service | Free tier | Link |
|---|---|---|
| **Gemini** | 15-30 req/min, 1000-1500/day | https://aistudio.google.com/apikey |
| **SerpAPI** | 100 searches/month | https://serpapi.com/manage-api-key |
| **Google Places (New)** | ~10 000 Text Search/month, requires a card on file | https://console.cloud.google.com — enable "Places API (New)" |

For Google Places, **set a budget alert at $1** in the Cloud Console (Billing → Budgets) as a safety net.

---

## Workflow

The pipeline is split into 8 steps, each presented as a section in the UI:

1. **Target company** — enter company name, country, search locale.
2. **Web search** — Gemini generates focused queries; SerpAPI runs them.
3. **Score & check links** — Gemini ranks the URLs; a pre-flight HTTP check detects unreachable links.
4. **Resolve unreachable links** — for each broken link, you can upload the PDF you've downloaded manually.
5. **Extract & structure data** — content is downloaded, chunked, and Gemini extracts slaughterhouses with strict separation between **capacity** (max potential) and **throughput** (actual annual figures). Cross-source synthesis deduplicates and enriches the list.
6. **Refine capacity** — optional targeted search per facility for missing capacities, with anti-hallucination guards (group-vs-facility scope check).
7. **Geocoding** — Google Places API as primary, Nominatim as fallback. Displays a map.
8. **Export** — final CSV ready for analysis.

---

## Tested companies

During development: SuKarne (Mexico), Bigard (France), Kilcoy Global Foods (Australia), Danish Crown (Denmark). Achieved 87% precision and 90% recall on a large-scale Bigard test (27 correct out of 30 identified, missing 3 small subsidiaries).
