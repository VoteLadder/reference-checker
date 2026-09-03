import os
import time
import json
import re
import requests
import logging
import shutil
from collections import defaultdict
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()



# ===========================
# Global Configuration and Constants
# ===========================
PDF_FILEPATH = 'article.pdf'
OUTPUT_DIR = 'ArticleV'
WORD_LIMIT = 12000


MAIN_API_KEY = os.getenv("MAIN_API_KEY")
MAIN_API_URL = os.getenv("MAIN_API_URL")
MAIN_MODEL = os.getenv("MAIN_MODEL", "deepseek/deepseek-v4-flash")

VERIF_API_KEY = os.getenv("VERIF_API_KEY")
VERIF_API_URL = os.getenv("VERIF_API_URL")
VERIF_MODEL = os.getenv("VERIF_MODEL", "deepseek/deepseek-v4-flash")



# Third party services
CROSSREF_API_URL   = "https://api.crossref.org/works"
UNPAYWALL_API_URL  = "https://api.unpaywall.org/v2/"
UNPAYWALL_EMAIL    = "john1221@yahoo.com"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ELINK_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
OPENALEX_API_URL     = "https://api.openalex.org/works"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Bounded network behavior for open-access fallback lookups/downloads
OA_METADATA_TIMEOUT  = 8    # seconds per metadata query (Crossref/Unpaywall/OpenAlex/EuropePMC)
PDF_DOWNLOAD_TIMEOUT = 20   # seconds per PDF download attempt
OA_MAX_PDF_URLS      = 4    # max candidate PDF URLs attempted per reference
MAX_PDF_BYTES        = 50 * 1024 * 1024  # 50 MB cap per downloaded PDF
MAX_FULLTEXT_BYTES   = 10 * 1024 * 1024  # 10 MB cap per Europe PMC XML source

# Request parameters
REQUESTS_TIMEOUT   = 1500
MAX_RETRIES        = 3
BACKOFF_FACTOR     = 2
INITIAL_BACKOFF    = 1
MAX_WORKERS        = 5

# ===========================
# Setup Logging and Session
# ===========================
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
session = requests.Session()

# ===========================
# Utility: Assign Unique IDs to Sentences
# ===========================
def assign_sentence_ids(sentences):
    for idx, sent in enumerate(sentences, start=1):
        sent["id"] = idx
    return sentences

# ===========================
# Text Extraction Functions
# ===========================
def extract_relevant_text(filepath, word_limit=50000):
    doc = fitz.open(filepath)
    words = []
    ref_patterns = [
        re.compile(r"\bReferences\b", re.IGNORECASE),
        re.compile(r"\bBibliography\b", re.IGNORECASE),
        re.compile(r"\bCitations\b", re.IGNORECASE)
    ]
    ref_found = False
    for page_index in range(len(doc)-1, -1, -1):
        page = doc[page_index]
        blocks = page.get_text("blocks")
        for block in blocks[::-1]:
            if not ref_found:
                for pat in ref_patterns:
                    if pat.search(block[4]):
                        ref_found = True
                        break
            if ref_found:
                if block[6] not in [1, 2]:
                    block_text = block[4].strip()
                    block_words = block_text.split()
                    words = block_words + words
                    if len(words) >= word_limit:
                        return ' '.join(words[:word_limit])
    return ' '.join(words)

def extract_references_section(filepath):
    doc = fitz.open(filepath)
    ref_text = ""
    ref_patterns = [
        re.compile(r"\bReferences\b", re.IGNORECASE),
        re.compile(r"\bBibliography\b", re.IGNORECASE),
        re.compile(r"\bCitations\b", re.IGNORECASE)
    ]
    ref_found = False
    for page_index in range(len(doc)):
        page = doc[page_index]
        blocks = page.get_text("blocks")
        for block in blocks:
            if not ref_found:
                for pat in ref_patterns:
                    if pat.search(block[4]):
                        ref_found = True
                        ref_text += block[4].strip() + "\n"
                        break
            else:
                ref_text += block[4].strip() + "\n"
    return ref_text.strip()

# ===========================
# Other Utility Functions
# ===========================
def clear_articles_directory(directory):
    d = os.path.join(directory, "articles")
    if os.path.exists(d):
        logger.debug(f"Clearing existing articles in {d} ...")
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    logger.debug("Articles directory cleared and reset.")

def robust_request(url, method="GET", headers=None, params=None, data=None, json_data=None, stream=False, timeout=REQUESTS_TIMEOUT, max_retries=MAX_RETRIES):
    backoff = INITIAL_BACKOFF
    headers = headers or {}
    headers.setdefault('User-Agent', 'ArticleAnalyzer/1.0')
    for attempt in range(max_retries):
        try:
            if method.upper() == "GET":
                response = session.get(url, headers=headers, params=params, stream=stream, timeout=timeout)
            elif method.upper() == "POST":
                response = session.post(url, headers=headers, params=params, data=data, json=json_data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            if response.status_code == 429:
                logger.debug(f"Rate limited for URL {url}. Waiting {backoff}s...")
                time.sleep(backoff)
                backoff *= BACKOFF_FACTOR
                continue
            elif response.status_code == 404:
                logger.debug(f"Resource not found: {url}")
                return None
            elif response.status_code >= 400:
                logger.debug(f"HTTP {response.status_code} for {url}")
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= BACKOFF_FACTOR
                    continue
                return None
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= BACKOFF_FACTOR
                continue
            return None
    return None

def clean_json_response(raw_text):
    try:
        txt = raw_text.strip()
        if txt.startswith("```json"):
            txt = txt[len("```json"):].strip()
        elif txt.startswith("```"):
            txt = txt[len("```"):].strip()
        if txt.endswith("```"):
            txt = txt[:-3].strip()
        start = txt.find('{')
        end = txt.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response.")
        content = txt[start:end+1]
        content = content.replace('\n', ' ').replace('\r', ' ')
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
        content = content.encode('ascii', 'ignore').decode('ascii')
        content = content.strip()
        json.loads(content)
        logger.debug(f"Cleaned JSON content: {content}")
        return content
    except Exception as e:
        logger.debug(f"Error in clean_json_response: {str(e)}\nProblematic text: {txt}")
        raise

def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        logger.debug(f"Error extracting text from {pdf_path}: {e}")
        return ""

# ===========================
# DOI and PDF Retrieval Functions
# ===========================
def get_doi_via_crossref(title, first_author, email=UNPAYWALL_EMAIL):
    params = {'query.title': title, 'query.author': first_author, 'rows': 1}
    headers = {'User-Agent': f'Provider/1.0 (mailto:{email})'}
    response = robust_request(CROSSREF_API_URL, headers=headers, params=params,
                              timeout=OA_METADATA_TIMEOUT, max_retries=1)
    if response:
        try:
            data = response.json()
            items = data.get('message', {}).get('items', [])
            if items:
                return items[0].get('DOI')
        except json.JSONDecodeError as e:
            logger.debug("CrossRef JSON decode error: %s", e)
    return None

def get_doi_from_reference(ref):
    doi = ref.get('doi')
    if doi:
        return doi
    title = ref.get('title', '')
    first_author = ref.get('first_author', '')
    doi = get_doi_via_crossref(title, first_author)
    if doi:
        ref['doi'] = doi
    return doi

def _dedupe_http_urls(urls):
    """Preserve order, drop blanks/non-http(s) URLs and duplicates."""
    seen = set()
    out = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u or not u.lower().startswith(("http://", "https://")):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _unpaywall_pdf_urls_from_record(data):
    """Collect viable OA PDF URLs from an Unpaywall record (best + all locations)."""
    candidates = []
    if not isinstance(data, dict):
        return candidates
    best = data.get('best_oa_location') or {}
    if isinstance(best, dict) and best.get('url_for_pdf'):
        candidates.append(best['url_for_pdf'])
    for loc in data.get('oa_locations') or []:
        if isinstance(loc, dict) and loc.get('url_for_pdf'):
            candidates.append(loc['url_for_pdf'])
    return _dedupe_http_urls(candidates)


def get_unpaywall_pdf_urls(doi):
    """Return ALL deduplicated Unpaywall OA PDF URLs for a DOI (best + oa_locations)."""
    if not doi:
        return []
    url = f"{UNPAYWALL_API_URL}{doi}"
    params = {'email': UNPAYWALL_EMAIL}
    response = robust_request(url, params=params, timeout=OA_METADATA_TIMEOUT,
                              max_retries=1)
    if response:
        try:
            data = response.json()
            if data.get('is_oa'):
                return _unpaywall_pdf_urls_from_record(data)
            logger.debug(f"Unpaywall record for {doi} is not open access")
        except json.JSONDecodeError as e:
            logger.debug("Unpaywall JSON decode error: %s", e)
    return []


def get_unpaywall_pdf_url(doi):
    urls = get_unpaywall_pdf_urls(doi)
    return urls[0] if urls else None


def get_openalex_pdf_urls(doi):
    """Query OpenAlex by DOI; collect/dedupe direct OA pdf_url values.

    Only uses openly licensed locations (best_oa_location plus locations
    flagged is_oa). Never scrapes publisher paywalls.
    """
    if not doi:
        return []
    url = f"{OPENALEX_API_URL}/https://doi.org/{doi}"
    params = {'mailto': UNPAYWALL_EMAIL}
    headers = {'User-Agent': f'Provider/1.0 (mailto:{UNPAYWALL_EMAIL})'}
    response = robust_request(url, headers=headers, params=params,
                              timeout=OA_METADATA_TIMEOUT, max_retries=1)
    if not response:
        return []
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.debug("OpenAlex JSON decode error: %s", e)
        return []
    candidates = []
    best = data.get('best_oa_location') or {}
    if isinstance(best, dict) and best.get('pdf_url'):
        candidates.append(best['pdf_url'])
    for loc in data.get('locations') or []:
        if isinstance(loc, dict) and loc.get('pdf_url') and loc.get('is_oa'):
            candidates.append(loc['pdf_url'])
    urls = _dedupe_http_urls(candidates)
    if urls:
        logger.debug(f"OpenAlex found {len(urls)} OA PDF URL(s) for DOI {doi}")
    return urls


def _normalize_pmcid(pmcid):
    """Normalize a PMCID value to digits-only (strip PMC prefix/whitespace)."""
    if pmcid is None:
        return ""
    s = str(pmcid).strip()
    if s.upper().startswith("PMC"):
        s = s[3:]
    return s.strip()


def get_europepmc_pmcid(doi):
    """Query Europe PMC by DOI; return digits-only PMCID for open-access records.

    Returns None unless the record is explicitly open access (isOpenAccess='Y')
    and carries a PMCID. The caller then uses the established PMC PDF URL.
    """
    if not doi:
        return None
    params = {'query': f'DOI:"{doi}"', 'format': 'json',
              'resultType': 'core', 'pageSize': 1}
    response = robust_request(EUROPEPMC_SEARCH_URL, params=params,
                              timeout=OA_METADATA_TIMEOUT, max_retries=1)
    if not response:
        return None
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.debug("Europe PMC JSON decode error: %s", e)
        return None
    try:
        results = (data.get('resultList') or {}).get('result') or []
    except AttributeError:
        return None
    if not results or not isinstance(results[0], dict):
        logger.debug(f"Europe PMC: no record for DOI {doi}")
        return None
    rec = results[0]
    if str(rec.get('isOpenAccess', '')).upper() != 'Y':
        logger.debug(f"Europe PMC record for DOI {doi} is not open access")
        return None
    pmcid = _normalize_pmcid(rec.get('pmcid', ''))
    if pmcid and pmcid.isdigit():
        return pmcid
    logger.debug(f"Europe PMC record for DOI {doi} has no PMCID")
    return None


def download_europepmc_fulltext(pmc_id, output_path):
    """Save official Europe PMC open-access full text as plain text.

    This runs only when an OA PDF is unavailable. It does not bypass publisher
    access controls and records Europe PMC XML provenance.
    """
    digits = _normalize_pmcid(pmc_id)
    if not digits or not digits.isdigit() or digits == "0":
        return False
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{digits}/fullTextXML"
    response = robust_request(url, timeout=OA_METADATA_TIMEOUT, max_retries=1)
    if not response:
        return False
    content = response.content
    if not content or len(content) > MAX_FULLTEXT_BYTES:
        logger.debug(f"Europe PMC XML missing or exceeds size cap for PMC{digits}")
        return False
    try:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(content)
        text = re.sub(r'\s+', ' ', ' '.join(root.itertext())).strip()
    except Exception as e:
        logger.debug(f"Europe PMC XML parse failed for PMC{digits}: {e}")
        return False
    if len(text) < 500:
        logger.debug(f"Europe PMC XML is too short for PMC{digits}")
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except OSError as e:
        logger.debug(f"Europe PMC XML save failed for PMC{digits}: {e}")
        return False


def _looks_like_pdf(first_chunk, content_type):
    ct = (content_type or '').lower()
    if 'application/pdf' in ct:
        return True
    return first_chunk[:4] == b'%PDF'


def download_pdf(pdf_url, output_path):
    """Download a PDF with strict validation and bounded network behavior.

    Only saves content that validates as a real PDF (Content-Type and/or
    %PDF magic bytes, plus a successful PyMuPDF open with >=1 page).
    Caps download size at MAX_PDF_BYTES. Returns True only for valid PDFs.
    """
    if not pdf_url or not str(pdf_url).lower().startswith(("http://", "https://")):
        logger.debug(f"Refusing to download from non-HTTP URL: {pdf_url}")
        return False
    headers = {"User-Agent": "Mozilla/5.0"}
    response = robust_request(pdf_url, headers=headers, stream=True,
                              timeout=PDF_DOWNLOAD_TIMEOUT, max_retries=1)
    if not response:
        logger.debug(f"Failed retrieving PDF from {pdf_url}")
        return False
    try:
        first = next(response.iter_content(1024), b'')
        ct = response.headers.get('Content-Type', '')
        if not _looks_like_pdf(first, ct):
            logger.debug(f"URL did not return PDF content (Content-Type={ct!r}): {pdf_url}")
            return False
        total = len(first)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(first)
            for chunk in response.iter_content(65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    logger.debug(f"PDF exceeds size cap ({MAX_PDF_BYTES} bytes): {pdf_url}")
                    break
                f.write(chunk)
        if total > MAX_PDF_BYTES:
            try:
                os.remove(output_path)
            except OSError:
                pass
            return False
        # Strict validation: must open as a real PDF with at least one page
        try:
            with fitz.open(output_path) as doc:
                if doc.page_count < 1:
                    raise ValueError("PDF has no pages")
        except Exception as e:
            logger.debug(f"Downloaded file failed PDF validation ({pdf_url}): {e}")
            try:
                os.remove(output_path)
            except OSError:
                pass
            return False
        logger.debug(f"Downloaded PDF: {output_path}")
        return True
    except Exception as e:
        logger.debug(f"Error saving PDF to {output_path}: {e}")
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass
    return False


def _try_pdf_candidates(ref, candidates, output_dir, ref_num):
    """Attempt each (pdf_url, download_source, doi_found) candidate in order.

    Sets ref provenance (download_source, web_address, doi_found) on success.
    Returns True on the first valid PDF download.
    """
    for pdf_url, source, doi_found in candidates[:OA_MAX_PDF_URLS]:
        pdf_filename = f"{ref_num}.pdf"
        pdf_path = os.path.join(output_dir, "articles", pdf_filename)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        if download_pdf(pdf_url, pdf_path):
            ref['doi_found'] = doi_found
            ref['download_source'] = source
            ref['download_format'] = 'pdf'
            ref['web_address'] = pdf_url
            return True
        logger.debug(f"{source} download failed for ref {ref_num}: {pdf_url}")
    return False

def get_article_pdf(ref, output_dir, ref_num):
    title = ref.get('title', '')
    first_author = ref.get('first_author', '')
    pmc_id = ref.get('pmc_id', '0')
    doi = get_doi_from_reference(ref)
    candidates = []
    seen = set()

    def _add(url, source, doi_found):
        if isinstance(url, str) and url not in seen:
            seen.add(url)
            candidates.append((url, source, doi_found))

    epub_pmcid = None
    if doi:
        # (1) All viable Unpaywall OA PDF URLs (best + oa_locations), deduped
        for u in get_unpaywall_pdf_urls(doi):
            _add(u, 'Unpaywall', True)
        if candidates:
            logger.debug(f"Unpaywall: {len(candidates)} OA candidate(s) for ref {ref_num}")
        else:
            logger.debug(f"No Unpaywall OA URL for DOI {doi} (ref {ref_num})")
        # (2) OpenAlex direct OA pdf_url values, deduped (skip Unpaywall dupes)
        for u in get_openalex_pdf_urls(doi):
            _add(u, 'OpenAlex', True)
        # (3) Europe PMC: resolve a PMCID only for confirmed OA records.
        epub_pmcid = get_europepmc_pmcid(doi)
        if epub_pmcid:
            if not pmc_id or str(pmc_id) in ["0", "null", ""]:
                ref['pmc_id'] = epub_pmcid
                pmc_id = epub_pmcid
    else:
        logger.debug(f"No DOI for ref {ref_num}")

    if candidates and _try_pdf_candidates(ref, candidates, output_dir, ref_num):
        return True

    # (4) Official Europe PMC XML is usable when no OA PDF succeeds.
    if epub_pmcid:
        txt_path = os.path.join(output_dir, "articles", f"{ref_num}.txt")
        if download_europepmc_fulltext(epub_pmcid, txt_path):
            ref['doi_found'] = bool(doi)
            ref['download_source'] = 'Europe PMC'
            ref['download_format'] = 'fulltext_xml'
            ref['web_address'] = f"https://europepmc.org/articles/PMC{epub_pmcid}"
            return True
    return False

def get_all_article_pdfs(references, output_dir):
    articles_dir = os.path.join(output_dir, "articles")
    os.makedirs(articles_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_article_pdf, ref, output_dir, ref_num): ref_num 
                   for ref_num, ref in references.items()}
        for fut in as_completed(futures):
            ref_num = futures[fut]
            try:
                result = fut.result()
                references[ref_num]['article_downloaded'] = result
                if result:
                    logger.debug(f"Downloaded article for ref {ref_num}")
                else:
                    logger.debug(f"Failed to download article for ref {ref_num}")
            except Exception as e:
                logger.debug(f"Exception during PDF download for ref {ref_num}: {e}")
                references[ref_num]['article_downloaded'] = False

def save_articles_zip(output_dir):
    zip_path = os.path.join(output_dir, "articles.zip")
    articles_path = os.path.join(output_dir, "articles")
    with ZipFile(zip_path, "w") as zipf:
        for filename in os.listdir(articles_path):
            if filename.endswith((".pdf", ".txt")):
                zipf.write(os.path.join(articles_path, filename), filename)
    logger.debug(f"Articles zipped into: {zip_path}")
    return zip_path

# ===========================
# Provider API Functions
# ===========================
class ProviderAPIError(RuntimeError):
    """Raised when the LLM provider call fails (HTTP error, network error, or bad payload)."""


def call_main_api(payload, api_url=MAIN_API_URL, api_key=MAIN_API_KEY, timeout=REQUESTS_TIMEOUT):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload.setdefault("model", MAIN_MODEL)
    payload.setdefault("reasoning", {"enabled": False})  # non-thinking mode
    try:
        response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 429:
            logger.debug("Rate limited at main API; backing off...")
            time.sleep(BACKOFF_FACTOR)
            response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("Main API HTTP %s: %s", response.status_code, response.text[:500])
            raise ProviderAPIError(f"Main API HTTP {response.status_code}: {response.text[:300]}") from e
        try:
            ret = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Main API returned non-JSON response: %s", response.text[:500])
            raise ProviderAPIError(f"Main API non-JSON response: {response.text[:300]}") from e
        logger.debug("Raw main API response: %s", ret)
        return ret
    except ProviderAPIError:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Main API call exception: {e}")
        raise ProviderAPIError(f"Main API request failed: {e}") from e

def call_verification_api(payload, api_url=VERIF_API_URL, api_key=VERIF_API_KEY, timeout=REQUESTS_TIMEOUT):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload.setdefault("model", VERIF_MODEL)
    payload.setdefault("reasoning", {"enabled": False})  # non-thinking mode
    payload.setdefault("temperature", 1)
    payload.setdefault("top_p", 1)
    payload.setdefault("repetition_penalty", 1)
    try:
        response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 429:
            logger.debug("Rate limited at verification API; backing off...")
            time.sleep(BACKOFF_FACTOR)
            response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("Verification API HTTP %s: %s", response.status_code, response.text[:500])
            raise ProviderAPIError(f"Verification API HTTP {response.status_code}: {response.text[:300]}") from e
        try:
            ret = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Verification API returned non-JSON response: %s", response.text[:500])
            raise ProviderAPIError(f"Verification API non-JSON response: {response.text[:300]}") from e
        logger.debug("Raw verification API response: %s", ret)
        return ret
    except ProviderAPIError:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Verification API call exception: {e}")
        raise ProviderAPIError(f"Verification API request failed: {e}") from e

# ===========================
# Main Content & References Processing
# ===========================
def process_main_content(text):
    prompt = (
        "Analyze the following article text and extract only the sentences that include citation references. "
        "DO NOT INCLUDE SENTENCES THAT DO NOT HAVE REFERENCES! "
        "Output a JSON object with this exact format:\n"
        '{"sentences": [{"sentence": "Full sentence text", "references": [1,2]}]}'
        "\nEnsure all text values use standard double quotes (\") and all property names are quoted."
    )
    content = f"[TEXT]{text}[/TEXT]\n\n{prompt}"
    payload = {"messages": [{"role": "user", "content": content}], "temperature": 0.0}
    try:
        logger.debug("Sending main content to main provider ...")
        response = call_main_api(payload)
        raw = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        cleaned = clean_json_response(raw)
        data = json.loads(cleaned)
        if not isinstance(data, dict) or "sentences" not in data:
            raise ValueError("Invalid JSON structure: missing 'sentences' key")
        sentences = []
        for sent in data["sentences"]:
            if isinstance(sent, dict) and "sentence" in sent and "references" in sent:
                sentences.append({
                    "sentence": str(sent["sentence"]),
                    "references": [int(ref) for ref in sent["references"]]
                })
        # Assign unique IDs to each sentence for later matching
        sentences = assign_sentence_ids(sentences)
        return {"sentences": sentences}
    except ProviderAPIError:
        logger.error("Main content extraction failed: provider API error; failing closed.")
        raise
    except Exception as e:
        logger.debug(f"Error processing main content: {str(e)}")
        return {"sentences": []}

def process_references_section(text):
    prompt = (
        "Extract all references from the following text. For each reference, extract the full title "
        "and the first author. Output a JSON object with this exact format:\n"
        '{"references": {"1": {"title": "Full Title", "first_author": "Author", '
        '"pmc_id": "0"}}, "total_references_extracted": 0}'
        "\nEnsure all text values use standard double quotes (\") and all property names are quoted."
    )
    content = f"[REFERENCES]{text}[/REFERENCES]\n\n{prompt}"
    payload = {"messages": [{"role": "user", "content": content}], "temperature": 0.0}
    try:
        logger.debug("Sending references section to main provider ...")
        response = call_main_api(payload)
        raw = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        cleaned = clean_json_response(raw)
        data = json.loads(cleaned)
        if not isinstance(data, dict) or "references" not in data:
            raise ValueError("Invalid JSON structure: missing 'references' key")
        references = {}
        for ref_id, ref in data["references"].items():
            if isinstance(ref, dict) and "title" in ref and "first_author" in ref:
                references[str(ref_id)] = {
                    "title": str(ref["title"]),
                    "first_author": str(ref["first_author"]),
                    "pmc_id": str(ref.get("pmc_id", "0"))
                }
        return {"references": references, "total_references_extracted": len(references)}
    except ProviderAPIError:
        logger.error("References extraction failed: provider API error; failing closed.")
        raise
    except Exception as e:
        logger.debug(f"Error processing references: {str(e)}")
        return {"references": {}, "total_references_extracted": 0}

# ===========================
# New Verification: Tagged Sentence Method
# ===========================
def verify_sentences_batch_tagged(article_text, tagged_claims):
    """
    Verify sentences that are tagged with an ID.
    tagged_claims: a list of dicts with keys "id" and "sentence".
    The prompt instructs the model to reply with each claim's id, a verdict (yes/no/maybe) and a 50-word explanation.
    """
    max_chars = 20000
    article_excerpt = article_text[:max_chars]
    claims_lines = []
    for claim in tagged_claims:
        claims_lines.append(f"Claim {claim['id']}: {claim['sentence']}")
    claims_text = "\n".join(claims_lines)
    prompt = (
        "Verify the following claims against the provided article excerpt. "
        "For each claim, respond with the claim ID, a verdict (yes/no/maybe) and a 50-word explanation. "
        "Output a JSON object exactly in this format:\n"
        '{"results": [ {"id": 1, "verdict": "yes/no/maybe", "explanation": "50 words explanation"}, ... ]}'
    )
    content = f"Article Excerpt:\n{article_excerpt}\n\nClaims to Verify:\n{claims_text}\n\n{prompt}"
    payload = {"messages": [{"role": "user", "content": content}], "temperature": 0.0}
    try:
        logger.debug("Sending batch verification request with tagged claims.")
        response = call_verification_api(payload)
        raw = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        cleaned = clean_json_response(raw)
        data = json.loads(cleaned)
        results = data.get("results", [])
        validated_results = []
        for result in results:
            if isinstance(result, dict) and all(k in result for k in ["id", "verdict", "explanation"]):
                validated_results.append({
                    "id": int(result["id"]),
                    "verdict": str(result["verdict"]).lower(),
                    "explanation": str(result["explanation"])
                })
        return validated_results
    except ProviderAPIError:
        logger.error("Batch verification failed: provider API error; failing closed.")
        raise
    except Exception as e:
        logger.debug(f"Error in verification (tagged): {str(e)}")
        return []

def process_articles_with_verification(articles_dir, sentences_data):
    """
    For each reference, send the tagged claims (with sentence IDs) for verification.
    Then update each sentence by matching the returned ID.
    """
    # Build mapping: ref_id -> list of tagged claims (with id and sentence)
    ref_map = defaultdict(list)
    for sent in sentences_data:
        for ref in sent.get("references", []):
            ref_map[str(ref)].append({"id": sent["id"], "sentence": sent["sentence"]})
    
    # Build an index to update sentences by ID
    id_index = {sent["id"]: sent for sent in sentences_data}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for ref_id, tagged_claims in ref_map.items():
            pdf_path = os.path.join(articles_dir, f"{ref_id}.pdf")
            text_path = os.path.join(articles_dir, f"{ref_id}.txt")
            if os.path.exists(pdf_path) or os.path.exists(text_path):
                try:
                    if os.path.exists(pdf_path):
                        article_text = extract_text_from_pdf(pdf_path)
                    else:
                        with open(text_path, encoding='utf-8') as f:
                            article_text = f.read()
                    if article_text:
                        future = executor.submit(verify_sentences_batch_tagged, article_text, tagged_claims)
                        future_map[future] = ref_id
                except Exception as e:
                    logger.debug(f"Error extracting source text for ref {ref_id}: {str(e)}")
            else:
                logger.debug(f"Article source for ref {ref_id} not found.")
        
        for future in as_completed(future_map):
            ref_id = future_map[future]
            try:
                results = future.result()
                for res in results:
                    sent_id = res["id"]
                    if sent_id in id_index:
                        id_index[sent_id].setdefault("verifications", []).append({
                            "reference": int(ref_id),
                            "verdict": res["verdict"],
                            "explanation": res["explanation"]
                        })
            except Exception as e:
                logger.debug(f"Error processing verification results for ref {ref_id}: {str(e)}")
                for claim in ref_map[ref_id]:
                    sent_id = claim["id"]
                    if sent_id in id_index:
                        id_index[sent_id].setdefault("verifications", []).append({
                            "reference": int(ref_id),
                            "verdict": "error",
                            "explanation": f"Verification failed: {str(e)}"
                        })
    return sentences_data

def extract_title_and_summary(text: str) -> tuple:
    """Extract title and summary using the verification API"""
    # Truncate to first 1000 words
    first_1000 = ' '.join(text.split()[:1000])
    
    prompt = (
        "Given the beginning of an academic article, extract its title and provide a 250-word summary. "
        "Output as JSON with this format:\n"
        '{"title": "The full title of the paper", "summary": "250-word summary"}'
    )
    
    content = f"[TEXT]{first_1000}[/TEXT]\n\n{prompt}"
    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "model": VERIF_MODEL
    }
    
    try:
        response = call_verification_api(payload)
        raw = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        cleaned = clean_json_response(raw)
        data = json.loads(cleaned)
        
        if not isinstance(data, dict) or "title" not in data or "summary" not in data:
            raise ValueError("Invalid JSON structure")
            
        return data["title"], data["summary"]
    except ProviderAPIError:
        logger.error("Title/summary extraction failed: provider API error; failing closed.")
        raise
    except Exception as e:
        logger.debug(f"Error extracting title and summary: {str(e)}")
        return None, None
# ===========================
# Main Function
# ===========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clear_articles_directory(OUTPUT_DIR)
    
    main_text = extract_relevant_text(PDF_FILEPATH, word_limit=WORD_LIMIT)
    references_text = extract_references_section(PDF_FILEPATH)
    
    if not main_text:
        logger.debug("Main text extraction failed.")
        return
    if not references_text:
        logger.debug("References section extraction failed.")
        return

    logger.debug("Extracted main content and references from PDF.")
    
    main_data = process_main_content(main_text)
    sentences_data = main_data.get("sentences", [])
    ref_data = process_references_section(references_text)
    references = ref_data.get("references", {})
    
    logger.debug("Resolving PMC IDs...")
    for key, ref in references.items():
        title = ref.get('title', '')
        first_author = ref.get('first_author', '')
        pmc = "0"  # Replace with actual lookup if available
        ref['pmc_id'] = str(pmc) if pmc else "0"
    
    logger.debug("Downloading article PDFs ...")
    get_all_article_pdfs(references, OUTPUT_DIR)
    
    logger.debug("Batch verifying sentences against article texts (tagged method)...")
    articles_dir = os.path.join(OUTPUT_DIR, "articles")
    sentences_data = process_articles_with_verification(articles_dir, sentences_data)
    
    final_output = {"sentences": sentences_data, "references": references}
    out_path = os.path.join(OUTPUT_DIR, "verified.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logger.debug(f"Final output saved to: {out_path}")
    except Exception as e:
        logger.debug(f"Output save error: {e}")
    
    confirmations_txt_path = os.path.join(OUTPUT_DIR, "confirmations.txt")
    try:
        with open(confirmations_txt_path, "w", encoding="utf-8") as f:
            f.write("Citation Confirmation Verifications\n")
            f.write("="*40 + "\n\n")
            for sent in final_output.get("sentences", []):
                if "verifications" in sent:
                    f.write("Sentence (ID {}): {}\n".format(sent["id"], sent["sentence"]))
                    for ver in sent["verifications"]:
                        f.write("  - Reference: {} | Verdict: {} | Explanation: {}\n".format(
                            ver.get("reference", ""),
                            ver.get("verdict", ""),
                            ver.get("explanation", "")
                        ))
                    f.write("-"*40 + "\n")
        logger.debug(f"Confirmation text file saved to: {confirmations_txt_path}")
    except Exception as e:
        logger.debug(f"Error writing confirmations text file: {e}")
    
    save_articles_zip(OUTPUT_DIR)

if __name__ == "__main__":
    main()

