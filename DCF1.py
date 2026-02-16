# VAL1_fixed_split_v3.py
# Fixed extractor v3:
# - Blend-first detection retained, but short-circuit / split rows only when a blend is confirmed.
# - Confirmation = global valuation_info['Blended'] OR blend phrase found near the PT.
# - Keeps price target same; emits two rows only when confirmed blended.
#
# Run: streamlit run VAL1_fixed_split_v3.py

import re
import io
from datetime import datetime
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta

import streamlit as st
import pandas as pd
import pdfplumber

# ---------------------------
# Tunables
# ---------------------------
DEBUG = False
SEARCH_RADIUS_CHARS = 600
YEAR_SEARCH_WINDOW = 1000  # Increased for better year capture

# ---------------------------
# Broker list
# ---------------------------
BROKERS = [
    "Goldman Sachs","Morgan Stanley","J.P. Morgan","JP Morgan","JPMorgan","Morning Star",
    "UBS","Barclays","Bank of America","BAML","Deutsche Bank","DB",
    "Credit Suisse","Citi Research","Citi","Citigroup","Jefferies","Evercore","RBC","Craig Hallum","OPCO",
    "Wells Fargo","Nomura","Piper Sandler","Roth","D.A. Davidson",
    "Rosenblatt","Macquarie","Canaccord","Cowen","Stifel","HSBC",
    "Berenberg","Bernstein","BNP Paribas","Societe Generale",
    "Mizuho","KeyBanc","Oppenheimer","SunTrust","Raymond James",
    "Telsey Advisory Group","Telsey","Guggenheim","Lazard","Morrison Foerster",
    "Moelis","PJT Partners","Sandler O'Neill","Sandler","BMO Capital Markets",
    "Scotiabank","TD Securities","Canaccord Genuity","Stifel Nicolaus",
    "FBN", "Wolfe Research", "BofA", "TD Cowen", "KBW", "RBC Capital Markets", "D.A. Davidson","Baird"
]
BROKERS_SORTED = sorted(BROKERS, key=lambda x: -len(x))

# ---------------------------
# Valuation canonical mapping
# ---------------------------
VALUATION_PATTERNS = {
    r'\bDCF\b': 'DCF',
    r'discounted\s+cash\s+flow': 'DCF',
    r'\bWACC\b': 'WACC',
    r'\bEV\s*\/?\s*EBITDA\b': 'EV/EBITDA',
    r'\bEV\s*\/?\s*FCF\b': 'EV/FCF',
    r'\bEV\s*\/?\s*SALES\b': 'EV/Sales',
    r'\bEV\s*\/?\s*REVENUE\b': 'EV/Revenue',
    r'\bP\s*\/?\s*E\b': 'P/E',
    r'\bP\s*\/?\s*S\b': 'P/Sales',
    r'\bPEG\b': 'PEG',
    r'\bSOTP\b': 'SOTP',
    r'sum\s+of\s+the\s+parts': 'SOTP',
    r'\bNAV\b': 'NAV',
    r'FCF\s*yield': 'FCF Yield'
}
VAL_PAT_TUPLES = [(re.compile(k, re.I), v) for k, v in VALUATION_PATTERNS.items()]

# ---------------------------
# DCF Parameters Patterns
# ---------------------------
DCF_PATTERNS = {
    'WACC': re.compile(r'\bWACC\b.*?(\d+(?:\.\d+)?)\s*%?', re.I),
    'Terminal Growth Rate': re.compile(r'terminal\s+(?:growth\s+)?rate.*?(\d+(?:\.\d+)?)\s*%?', re.I),
    'Equity Risk Premium': re.compile(r'(?:equity\s+)?risk\s+premium.*?(\d+(?:\.\d+)?)\s*%?', re.I),
    'Risk Free Rate': re.compile(r'risk\s+free\s+(?:rate).*?(\d+(?:\.\d+)?)\s*%?', re.I),
    'FCF Estimates': re.compile(r'FY[\s\-]?(\d{2})\s*:?\s*\$?([,\d]+\.?\d*)', re.I),
    'Discount Period': re.compile(r'(?:\d+-year|\d+ year)\s+(?:explicit|forecast|projection)', re.I),
    'Enterprise Value': re.compile(r'enterprise\s+value.*?\$?([,\d]+\.?\d*)', re.I)
}

# ---------------------------
# Regex patterns
# ---------------------------
PRICE_TARGET_RE = re.compile(r'''(?xi)
    (?:
        (?:
            target\s*price|price\s*target|target\s*:|target\s*price(?:\s*/\s*base\s*case)?|
            target\s*[:\-\u2013\u2014]|\bPT\b|12\s*[-–]?\s*m(?:o|onth)?(?:\s*price\s*target)?
        )
        [\s:\-–—]*
        ([A-Z]{0,3}[\$\€\£\¥]?)?
        \s*(?:USD|US\$|CAD|EUR|GBP|INR|JPY)?\s*
        ([0-9]{1,5}(?:[,\.][0-9]{3})*(?:\.[0-9]+)?)
    )
    |
    (?:
        ([\$\€\£\¥])\s*([0-9]{1,5}(?:[,\.][0-9]{3})*(?:\.[0-9]+)?)\s*(?:target\s*price|price\s*target)
    )
''', re.MULTILINE | re.DOTALL)

PRICE_SENTENCE_RE = re.compile(
    r'(?i)(?:sets target at|sets a target of|raises target to|lowers target to|new target|target(?:\s*price)?)\s*([A-Z]{0,3}[\$\€\£\¥]?)?\s*([0-9]{1,5}(?:[,\.][0-9]{3})*(?:\.[0-9]+)?)'
)

PRICE_TABLE_RE = re.compile(
    r'(?i)(?:target\s*price|price\s*target)[^\dA-Z\$\€\£\¥]{0,20}([A-Z]{0,3}[\$\€\£\¥]?)?\s*([0-9]{1,5}(?:[,\.][0-9]{3})*(?:\.[0-9]+)?)',
    re.MULTILINE | re.DOTALL
)

MULTIPLE_PATTERN = re.compile(
    r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\s*(EV\s*\/?\s*FCF|EV\s*\/?\s*EBITDA|EV\s*\/?\s*SALES|EV\s*\/?\s*REVENUE|P\s*\/?\s*E|P\s*\/?\s*S|PEG|FCF|EBITDA|SALES|REVENUE)',
    re.I
)
LOOSE_MULT_RE = re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\b', re.I)

FUZZY_EV_FCF = re.compile(r'EV[\s\/\-\_\w]{0,20}FCF', re.I)
FUZZY_EV_SALES = re.compile(r'EV[\s\/\-\_\w]{0,20}SALES|EV[\s\/\-\_\w]{0,20}REVENUE|EV\s*\/\s*S\b', re.I)
FUZZY_EV_EBITDA = re.compile(r'EV[\s\/\-\_\w]{0,20}EBITDA', re.I)

RATIO_LIKE = re.compile(r'\b\d{1,3}\s*[x×X]\b|\b\d{1,3}\s*%\b|\b\d{1,3}\s*\/\s*\d{1,3}\b', re.I)
VALUATION_KEYWORDS_RE = re.compile(r'(?i)\b(EV|FCF|EBITDA|REVENUE|SALES|P/E|DCF|WACC|FCF yield)\b')
BLEND_TERMS = re.compile(r'(?i)\b(blend|blended|average|equal weight|equally[-\s]*weighted|weighted|50/50|50-50|50:50|60/40|33/33|mix|weighting)\b')

YEAR_4_RE = re.compile(r'\b(19|20)\d{2}\b')
FY_RE = re.compile(r'\bFY[\s\-]?(\d{2,4})[Ee]?\b', re.I)
CY_RE = re.compile(r'\bCY[\s\-]?(\d{2,4})[Ee]?\b', re.I)
TWELVE_M_RE = re.compile(r'\b12[-\s]?m(?:onth)?\b', re.I)

REPORT_DATE_PATTERNS = [
    r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}',
    r'\d{4}-\d{2}-\d{2}',
    r'\d{1,2}/\d{1,2}/\d{2,4}'
]

PRICE_GOOD_CTX = re.compile(r'(?i)\b(target price|price objective|price target|valuation methodology|target:|target price:|our target|base case)\b')
PRICE_BAD_CTX = re.compile(r'(?i)\b(current price|trading at|last close|previous close|as of|market price|price \(as of|price as of)\b')

# ---------------------------
# Utilities
# ---------------------------
def parse_number(s):
    if s is None:
        return None
    s = str(s).strip().replace(',', '')
    s = re.sub(r'^[^\d\-]+', '', s)
    try:
        return float(s)
    except Exception:
        return None

def extract_text(file_bytes):
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n\n".join(pages), pages

def normalize(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()

# ---------------------------
# Broker detection
# ---------------------------
def detect_broker(full_text, pages):
    txt = (full_text or "").lower()
    first_page = pages[0] if pages else full_text or ""
    first_block = first_page[:1200].lower()
    for b in BROKERS_SORTED:
        if re.search(r'\b' + re.escape(b.lower()) + r'\b', first_block):
            return b
    earliest = None
    earliest_broker = None
    for b in BROKERS_SORTED:
        m = re.search(r'\b' + re.escape(b.lower()) + r'\b', txt)
        if m:
            if earliest is None or m.start() < earliest:
                earliest = m.start()
                earliest_broker = b
    if earliest_broker:
        return earliest_broker
    tokens_text = set(re.findall(r'\b[a-z0-9]{2,}\b', txt))
    for b in BROKERS_SORTED:
        tokens = re.findall(r'\b[a-z0-9]{2,}\b', b.lower())
        if not tokens:
            continue
        matches = sum(1 for t in tokens if t in tokens_text)
        if matches >= min(2, len(tokens)):
            return b
    lines = [ln.strip() for ln in (first_page or "").splitlines() if ln.strip()]
    return lines[0] if lines else "Unknown"

# ---------------------------
# Year inference for PT and multiples
# ---------------------------
def infer_year_label(item, full_text, report_date):
    start = item.get("start", 0)
    end = item.get("end", 0)
    window = full_text[max(0, start - YEAR_SEARCH_WINDOW): min(len(full_text), end + YEAR_SEARCH_WINDOW)].upper()
    m = FY_RE.search(window)
    if m:
        yy = int(m.group(1)); year = 2000 + yy if yy < 100 else yy
        return f"FY {year}"
    m = CY_RE.search(window)
    if m:
        yy = int(m.group(1)); year = 2000 + yy if yy < 100 else yy
        return f"CY {year}"
    if TWELVE_M_RE.search(window) and report_date:
        year = (report_date + relativedelta(months=12)).year
        return f"12M {year}"
    m18 = re.search(r'\b18[\s\-]?(?:m|month)\b', window, re.I)
    if m18 and report_date:
        year = (report_date + relativedelta(months=18)).year
        return f"18M {year}"
    m = YEAR_4_RE.search(window)
    if m:
        return f"CY {int(m.group(0))}"
    if report_date:
        return f"CY {report_date.year}"
    return f"CY {datetime.now().year}"

# ---------------------------
# Extract DCF Parameters
# ---------------------------
def extract_dcf_parameters(valuation_text):
    dcf_params = {}
    for key, pattern in DCF_PATTERNS.items():
        matches = pattern.findall(valuation_text)
        if matches:
            if key == 'FCF Estimates':
                dcf_params[key] = '; '.join([f"FY{yy}: ${val}" for yy, val in matches])
            elif key == 'Discount Period':
                dcf_params[key] = matches[0] if matches else None
            else:
                dcf_params[key] = matches[0]  # Take the first match
        else:
            dcf_params[key] = None
    return dcf_params

# ---------------------------
# Price target extraction
# ---------------------------
def extract_price_targets_with_pos(text):
    pts = []
    text_norm = re.sub(r'\s+', ' ', text)

    anchors = list(re.finditer(r'(?i)\b(target price|price target|price objective|target:|our target|base case)\b', text_norm))
    anchor_candidates = []
    for a in anchors:
        s, e = a.span()
        buf_start = max(0, s - 400)
        buf_end = min(len(text_norm), e + 400)
        buf = text_norm[buf_start:buf_end]
        for m in re.finditer(r'([\$\€\£\¥]?\s*[0-9]{1,5}(?:[,\.][0-9]{3})*(?:\.[0-9]+)?)', buf):
            raw = m.group(1)
            val = parse_number(raw)
            if val is None or val < 5 or val > 5000:
                continue
            abs_start = buf_start + m.start()
            abs_end = buf_start + m.end()
            ctx = text_norm[max(0, abs_start - 150): min(len(text_norm), abs_end + 150)]
            score = 0
            if re.search(r'[\$\€\£\¥]|USD|US\$', raw, re.I): score += 5
            if PRICE_GOOD_CTX.search(ctx): score += 6
            if PRICE_BAD_CTX.search(ctx): score -= 8
            if re.search(r'\btarget\b', ctx, re.I): score += 2
            anchor_center = (s + e) / 2
            dist = abs(((abs_start + abs_end) / 2) - anchor_center)
            score -= dist / 1000.0
            anchor_candidates.append({
                'value': val,
                'currency_raw': raw.strip(),
                'currency': re.search(r'[\$\€\£\¥]', raw).group(0) if re.search(r'[\$\€\£\¥]', raw) else "",
                'start': abs_start,
                'end': abs_end,
                'context': ctx,
                'score': round(score, 3),
                'anchor_dist': dist
            })
    if anchor_candidates:
        anchor_candidates.sort(key=lambda x: (-x['score'], x['anchor_dist'], -x['value']))
        top = anchor_candidates[0]
        pts.append({
            'currency': top.get('currency'),
            'value': top.get('value'),
            'context': top.get('context'),
            'start': top.get('start'),
            'end': top.get('end')
        })
        return pts

    candidates = []
    for m in PRICE_TARGET_RE.finditer(text):
        candidates.append((m.start(), m.group(0), m.groups(), m.start(), m.end()))
    for m in PRICE_SENTENCE_RE.finditer(text):
        candidates.append((m.start(), m.group(0), (m.group(1), m.group(2)), m.start(), m.end()))
    for m in PRICE_TABLE_RE.finditer(text):
        candidates.append((m.start(), m.group(0), m.groups(), m.start(), m.end()))

    target_candidates = [c for c in candidates if re.search(r'\btarget\b', (c[1] or ''), re.I)]
    use_list = target_candidates if target_candidates else candidates
    use_list.sort(key=lambda x: x[0])

    scored = []
    for _, snippet, groups, start, end in use_list:
        currency = ""
        val = None
        for g in groups:
            if not g: continue
            if re.search(r'[\$\€\£\¥]|USD|US\$', str(g), re.I): currency = str(g).strip()
            if re.search(r'\d', str(g)):
                v = parse_number(g)
                if v is not None: val = v
        if not val: continue
        ctx = text[max(0, start - 150): min(len(text), end + 150)]
        if RATIO_LIKE.search(snippet): continue
        if val < 5 or val > 5000: continue
        score = 0
        if currency: score += 5
        if PRICE_GOOD_CTX.search(ctx): score += 6
        if PRICE_BAD_CTX.search(ctx): score -= 8
        if re.search(r'\btarget\b', snippet, re.I): score += 3
        if re.search(r'\b\d{1,3}\s*[x×X]\b', ctx): score -= 4
        scored.append({'currency': currency, 'value': val, 'context': snippet.strip(), 'start': start, 'end': end, 'score': score})

    if not scored:
        return []
    scored.sort(key=lambda x: (-x['score'], x['start']))
    best = scored[0]
    return [{'currency': best.get('currency'), 'value': best.get('value'), 'context': best.get('context'), 'start': best.get('start'), 'end': best.get('end')}]

# ---------------------------
# Multiples extraction & helpers
# ---------------------------
def detect_canonical_from_context(ctx):
    if not ctx:
        return None
    for patt, canon in VAL_PAT_TUPLES:
        if patt.search(ctx):
            return canon
    if re.search(r'\bfcf\s*yield\b', ctx, re.I):
        return 'FCF Yield'
    if FUZZY_EV_FCF.search(ctx):
        return 'EV/FCF'
    if FUZZY_EV_EBITDA.search(ctx):
        return 'EV/EBITDA'
    if FUZZY_EV_SALES.search(ctx):
        return 'EV/Sales'
    return None

def extract_all_multiples(full_text, pages, report_date):
    page_starts = []
    running = 0
    sep = "\n\n"
    for i, p in enumerate(pages):
        page_starts.append(running)
        running += len(p or "") + len(sep)

    valuation_page_indices = [i for i, p in enumerate(pages) if re.search(
        r'(?i)\b(valuation|valuation methodology|price target calculation|investment thesis|valuation and risks|price objective|valuation methodology)\b', p or "")]

    multiples = []
    # valuation pages scanning
    for pi in valuation_page_indices:
        page_text = pages[pi] or ""
        page_start_abs = page_starts[pi]
        for m in MULTIPLE_PATTERN.finditer(page_text):
            s = page_start_abs + m.start(); e = page_start_abs + m.end()
            num = parse_number(m.group(1)); label = (m.group(2) or "").strip()
            if num is None or not (1.5 <= num <= 200): continue
            canonical = None
            if re.search(r'EV\s*\/?\s*FCF', label, re.I) or ('FCF' in label.upper() and 'EV' in label.upper()):
                canonical = 'EV/FCF'
            elif re.search(r'EV\s*\/?\s*EBITDA', label, re.I): canonical = 'EV/EBITDA'
            elif re.search(r'EV\s*\/?\s*SALES|EV\s*\/?\s*REVENUE', label, re.I): canonical = 'EV/Sales'
            elif re.search(r'P\s*\/?\s*E', label, re.I): canonical = 'P/E'
            elif re.search(r'FCF\s*yield', label, re.I): canonical = 'FCF Yield'
            ctx = normalize(page_text[max(0, m.start()-80): m.end()+80])
            item = {'start': s, 'end': e}
            year_label = infer_year_label(item, full_text, report_date)
            multiples.append({'value': num, 'label': label if label else None, 'canonical': canonical, 'start': s, 'end': e, 'source': 'valuation_pages', 'context': ctx, 'year_label': year_label})

        for m in LOOSE_MULT_RE.finditer(page_text):
            s = page_start_abs + m.start(); e = page_start_abs + m.end()
            num = parse_number(m.group(1))
            if num is None or not (1.5 <= num <= 200): continue
            ctx_near = page_text[max(0, m.start()-160): m.end()+160]
            canonical = detect_canonical_from_context(ctx_near)
            label = canonical.replace('/', '') if canonical else None
            if any(abs(num - mm['value']) < 1e-6 and abs(s - mm['start']) < 8 for mm in multiples if mm['source']=='valuation_pages'):
                continue
            item = {'start': s, 'end': e}
            year_label = infer_year_label(item, full_text, report_date)
            multiples.append({'value': num, 'label': label, 'canonical': canonical, 'start': s, 'end': e, 'source': 'valuation_pages', 'context': normalize(ctx_near), 'year_label': year_label})

    # full text scanning
    for m in MULTIPLE_PATTERN.finditer(full_text):
        s = m.start(); e = m.end()
        num = parse_number(m.group(1)); label = (m.group(2) or "").strip()
        if num is None or not (1.5 <= num <= 200): continue
        canonical = None
        if re.search(r'EV\s*\/?\s*FCF', label, re.I) or ('FCF' in label.upper() and 'EV' in label.upper()):
            canonical = 'EV/FCF'
        elif re.search(r'EV\s*\/?\s*EBITDA', label, re.I): canonical = 'EV/EBITDA'
        elif re.search(r'EV\s*\/?\s*SALES|EV\s*\/?\s*REVENUE', label, re.I): canonical = 'EV/Sales'
        elif re.search(r'P\s*\/?\s*E', label, re.I): canonical = 'P/E'
        elif re.search(r'FCF\s*yield', label, re.I): canonical = 'FCF Yield'
        ctx = normalize(full_text[max(0, m.start()-80): m.end()+80])
        item = {'start': s, 'end': e}
        year_label = infer_year_label(item, full_text, report_date)
        multiples.append({'value': num, 'label': label if label else None, 'canonical': canonical, 'start': s, 'end': e, 'source': 'full_text', 'context': ctx, 'year_label': year_label})

    for m in LOOSE_MULT_RE.finditer(full_text):
        s = m.start(); e = m.end()
        num = parse_number(m.group(1))
        if num is None or not (1.5 <= num <= 200): continue
        ctx_near = full_text[max(0, m.start()-160): m.end()+160]
        canonical = detect_canonical_from_context(ctx_near)
        label = canonical.replace('/', '') if canonical else None
        if any(abs(num - mm['value']) < 1e-6 and abs(s - mm['start']) < 8 for mm in multiples):
            continue
        item = {'start': s, 'end': e}
        year_label = infer_year_label(item, full_text, report_date)
        multiples.append({'value': num, 'label': label, 'canonical': canonical, 'start': s, 'end': e, 'source': 'full_text', 'context': normalize(ctx_near), 'year_label': year_label})

    # dedupe
    dedup = []
    seen = set()
    for m in multiples:
        key = (round(m['value'], 4), m['start'])
        if key in seen: continue
        seen.add(key); dedup.append(m)
    dedup.sort(key=lambda x: x['start'])
    return dedup

# ---------------------------
# Valuation methods extraction
# ---------------------------
def extract_valuation_methods(full_text, pages):
    valuation_text = ""
    for p in pages:
        if re.search(r'(?i)\b(valuation|valuation methodology|price target calculation|investment thesis|valuation and risks|price objective|valuation methodology)\b', p or ""):
            valuation_text += p + "\n\n"
    if not valuation_text:
        valuation_text = full_text

    found = set()
    for patt, canon in VAL_PAT_TUPLES:
        if patt.search(valuation_text):
            found.add(canon)
    if FUZZY_EV_FCF.search(valuation_text):
        found.add('EV/FCF')
    if FUZZY_EV_EBITDA.search(valuation_text):
        found.add('EV/EBITDA')
    if FUZZY_EV_SALES.search(valuation_text):
        found.add('EV/Sales')
    if re.search(r'\bfcf\s*yield\b', valuation_text, re.I):
        found.add('FCF Yield')

    blended = False
    weights = []
    if BLEND_TERMS.search(valuation_text):
        blended = True
        raw_weights = re.findall(r'(\d{1,3}\s*\/\s*\d{1,3})', valuation_text)
        weights = [w for w in raw_weights if not re.match(r'^(0?[1-9]|1[0-2])\s*\/\s*(?:0?[1-9]|[12][0-9]|3[01])$', w)]

    methods_str = '; '.join(sorted(found)) if found else None
    confidence = 'High' if found else 'Low'

    dcf_present = 'DCF' in found
    dcf_params = extract_dcf_parameters(valuation_text) if dcf_present else {}

    return {
        'Valuation_Methods': methods_str,
        'Blended': blended,
        'Weights': weights,
        'Confidence': confidence,
        'DCF_Present': dcf_present,
        **dcf_params  # Add DCF params as additional keys
    }

# ---------------------------
# Helper: build_blended_values (robust)
# ---------------------------
def build_blended_values(pt, all_multiples, full_text, window=600):
    """
    Return (valuation_multiple_used, valuation_multiple_value) for EV/Sales + EV/FCF
    if both found near pt. Otherwise (None, None).
    """
    ev_sales_val = None
    ev_fcf_val = None
    text = full_text or ""
    text_len = len(text)

    start_win = max(0, pt.get('start', 0) - window)
    end_win   = min(text_len, pt.get('end', text_len) + window)
    win_text = text[start_win:end_win]

    # --- 1) High-priority: explicit "blend" phrase extractor ---
    blend_phrase_re = re.compile(r'(?i)(blend(?:ed)?\s*(?:of)?|equally[-\s]*weighted|equal\s*weight|50\s*\/\s*50|50-50|50:50).{0,300}', re.I | re.S)
    pair_re = re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\s*(EV[\s\/\-\_\w]{0,40}SALES|EV[\s\/\-\_\w]{0,40}REVENUE|EV[\s\/\-\_\w]{0,40}FCF|EV\s*\/\s*S\b|EV\s*\/\s*FCF\b)', re.I)
    label_then_num_re = re.compile(r'(EV[\s\/\-\_\w]{0,40}SALES|EV[\s\/\-\_\w]{0,40}REVENUE|EV[\s\/\-\_\w]{0,40}FCF|EV\s*\/\s*S\b|EV\s*\/\s*FCF\b).{0,60}?([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]', re.I)

    for bmatch in blend_phrase_re.finditer(win_text):
        start_local = bmatch.start()
        local = win_text[max(0, start_local - 40): min(len(win_text), start_local + 340)]
        sales_found = None
        fcf_found = None
        for pr in pair_re.finditer(local):
            num = pr.group(1)
            lbl = pr.group(2).upper()
            vtxt = f"{num}x"
            if re.search(r'FCF', lbl, re.I):
                if not fcf_found: fcf_found = vtxt
            if re.search(r'SALES|REVENUE|EV\s*\/\s*S', lbl, re.I):
                if not sales_found: sales_found = vtxt
        if not (sales_found and fcf_found):
            for pr in label_then_num_re.finditer(local):
                lbl = pr.group(1).upper(); num = pr.group(2); vtxt = f"{num}x"
                if re.search(r'FCF', lbl, re.I):
                    if not fcf_found: fcf_found = vtxt
                if re.search(r'SALES|REVENUE|EV\s*\/\s*S', lbl, re.I):
                    if not sales_found: sales_found = vtxt
        if sales_found and fcf_found:
            return "EV/Sales + EV/FCF", f"{sales_found} + {fcf_found}"

    # --- 2) Fast pass: use canonical / inferred canonical on extracted multiples ---
    for m in all_multiples:
        if m.get('start') is None or m.get('end') is None:
            continue
        if not (m['start'] >= start_win and m['end'] <= end_win):
            continue

        canon = m.get('canonical')
        if not canon:
            ctx = m.get('context') or text[max(0, m['start'] - 80): m['end'] + 80]
            canon = detect_canonical_from_context(ctx)
            lab = (m.get('label') or "").upper()
            if not canon:
                if re.search(r'EV\s*\/\s*S|EV\s*\/\s*SALES|EV\s*\/\s*REV', lab, re.I) or re.search(r'(?i)\bEV\s*\/\s*S\b', ctx):
                    canon = 'EV/Sales'
                if re.search(r'(?i)EV[\s\/\-\_\w]{0,20}FCF', ctx) or 'FCF' in lab:
                    canon = 'EV/FCF'

        val_txt = None
        if m.get('value') is not None:
            try:
                val_txt = f"{m.get('value')}x"
            except Exception:
                val_txt = str(m.get('value'))

        if (canon == 'EV/Sales' or m.get('inferred_canonical') == 'EV/Sales') and ev_sales_val is None and val_txt:
            ev_sales_val = val_txt
        if (canon == 'EV/FCF' or m.get('inferred_canonical') == 'EV/FCF') and ev_fcf_val is None and val_txt:
            ev_fcf_val = val_txt

        if ev_sales_val and ev_fcf_val:
            return "EV/Sales + EV/FCF", f"{ev_sales_val} + {ev_fcf_val}"

    # --- 3) Fallback: parse numbers in window and map by proximity/order ---
    nums = []
    for nm in re.finditer(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\b', win_text):
        raw = nm.group(1)
        pos_in_win = nm.start()
        abs_pos = start_win + pos_in_win
        try:
            _ = float(raw)
        except Exception:
            continue
        nums.append((abs_pos, f"{raw}x"))

    if not nums:
        return None, None

    revenue_keys = [m.start() + start_win for m in re.finditer(r'(?i)\b(revenue|sales|rev(?:enue)?)\b', win_text)]
    fcf_keys     = [m.start() + start_win for m in re.finditer(r'(?i)\b(free cash flow|fcf)\b', win_text)]

    def nearest_dist(pos, key_list):
        if not key_list:
            return None
        return min(abs(pos - k) for k in key_list)

    if revenue_keys and fcf_keys and len(nums) >= 2:
        scored = []
        for pos, vtxt in nums:
            d_rev = nearest_dist(pos, revenue_keys) or 10**6
            d_fcf = nearest_dist(pos, fcf_keys) or 10**6
            scored.append({'pos': pos, 'val': vtxt, 'd_rev': d_rev, 'd_fcf': d_fcf})
        rev_candidates = [s for s in scored if s['d_rev'] < s['d_fcf']]
        fcf_candidates = [s for s in scored if s['d_fcf'] < s['d_rev']]
        if rev_candidates and fcf_candidates:
            rev_choice = min(rev_candidates, key=lambda x: x['d_rev'])
            fcf_choice = min(fcf_candidates, key=lambda x: x['d_fcf'])
            return "EV/Sales + EV/FCF", f"{rev_choice['val']} + {fcf_choice['val']}"

    sents = re.split(r'(?<=[\.\;\:\n])\s+', win_text)
    for s in sents:
        if re.search(r'(?i)(revenue|sales|rev)', s) and re.search(r'(?i)\b(free cash flow|fcf)\b', s):
            local_nums = [(m.start(), m.group(1)) for m in re.finditer(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\b', s)]
            if len(local_nums) >= 2:
                ev_sales_val = f"{local_nums[0][1]}x"
                ev_fcf_val   = f"{local_nums[1][1]}x"
                return "EV/Sales + EV/FCF", f"{ev_sales_val} + {ev_fcf_val}"

    if revenue_keys:
        best = None; bestd = 10**6
        for pos, vtxt in nums:
            d = nearest_dist(pos, revenue_keys) or 10**6
            if d < bestd:
                bestd = d; best = vtxt
        if best:
            ev_sales_val = best
    if fcf_keys:
        best = None; bestd = 10**6
        for pos, vtxt in nums:
            d = nearest_dist(pos, fcf_keys) or 10**6
            if d < bestd:
                bestd = d; best = vtxt
        if best:
            ev_fcf_val = best

    if ev_sales_val and ev_fcf_val:
        return "EV/Sales + EV/FCF", f"{ev_sales_val} + {ev_fcf_val}"
    return None, None

# ---------------------------
# NEW helper: append records either split or single (split only when blended_flag True)
# ---------------------------
def append_split_records(records, base_record, vm_used, vm_value, weights, blended_flag, confidence, dcf_params):
    """
    If blended_flag is True AND vm_used contains a '+' and vm_value contains a '+',
    create two rows (one per method) preserving PT and other base fields.
    Otherwise append one row.
    """
    if blended_flag and vm_used and '+' in vm_used and vm_value and '+' in vm_value:
        parts_used = [p.strip() for p in vm_used.split('+')]
        parts_val = [v.strip() for v in vm_value.split('+')]
        for pu, pv in zip(parts_used, parts_val):
            r = base_record.copy()
            r['Valuation Multiple Used'] = pu
            r['Valuation Multiple Value'] = pv
            r['Weights'] = weights
            r['Blended'] = True
            r['Confidence'] = confidence
            # Add DCF params to each split row
            for key, val in dcf_params.items():
                r[key] = val
            records.append(r)
    else:
        r = base_record.copy()
        r['Valuation Multiple Used'] = vm_used
        r['Valuation Multiple Value'] = vm_value
        r['Weights'] = weights
        r['Blended'] = blended_flag
        r['Confidence'] = confidence
        # Add DCF params
        for key, val in dcf_params.items():
            r[key] = val
        records.append(r)

# ---------------------------
# Main Streamlit UI & Processing
# ---------------------------
st.set_page_config(page_title="VAL1 Fixed Split v3", layout="wide")
st.title("Sell-side PDF Extractor — VAL1_fixed_split_v3")

uploaded_files = st.file_uploader("Upload one or more PDFs", type=['pdf'], accept_multiple_files=True)
all_records = []

if uploaded_files:
    for uploaded in uploaded_files:
        file_bytes = uploaded.read()
        full_text, pages = extract_text(file_bytes)

        # detect report date from first page
        report_date = None
        first_page = pages[0] if pages else ''
        for pat in REPORT_DATE_PATTERNS:
            m = re.search(pat, first_page, re.I)
            if m:
                try:
                    report_date = dateparser.parse(m.group(0)); break
                except Exception:
                    pass

        broker = detect_broker(full_text, pages)
        pts = extract_price_targets_with_pos(full_text)
        valuation_info = extract_valuation_methods(full_text, pages)
        all_multiples = extract_all_multiples(full_text, pages, report_date)

        # debug expanders
        if DEBUG:
            with st.expander(f"Debug — All multiples discovered ({uploaded.name})"):
                if all_multiples:
                    st.write(pd.DataFrame(all_multiples))
                else:
                    st.write("No multiples discovered by patterns.")
            with st.expander(f"Debug — Valuation summary ({uploaded.name})"):
                st.write(valuation_info)

        dcf_params = {k: valuation_info.get(k, None) for k in DCF_PATTERNS.keys()}

        # ----- if no price targets found, doc-level blended detection (only split if valuation_info indicates blended)
        if not pts:
            # build doc-level pseudo-pt
            doc_pt = {'start': 0, 'end': len(full_text or "")}
            year_label = infer_year_label(doc_pt, full_text, report_date)
            vm_used, vm_value = build_blended_values(doc_pt, all_multiples, full_text)

            base = {
                'File': uploaded.name,
                'Broker': broker,
                'Price Target': None,
                'Currency': None,
                'Valuation Multiple Used': None,
                'Valuation Multiple Value': None,
                'Weights': None,
                'Blended': None,
                'Confidence': None,
                'Year': year_label,
                'Context': None,
                'File Date': report_date.date() if report_date else None,
                'DCF_Present': valuation_info.get('DCF_Present', False)
            }
            # Only split if valuation_info signals blended (doc-level)
            append_split_records(all_records, base, vm_used or valuation_info.get('Valuation_Methods'), vm_value, valuation_info.get('Weights'), valuation_info.get('Blended'), valuation_info.get('Confidence'), dcf_params)

        else:
            for pt in pts:
                year_label_for_pt = infer_year_label(pt, full_text, report_date)
                pt_mid = (pt['start'] + pt['end']) / 2
                win_start = max(0, pt['start'] - SEARCH_RADIUS_CHARS)
                win_end = min(len(full_text), pt['end'] + SEARCH_RADIUS_CHARS)

                # ---------- BLEND-FIRST: detect blended pair, but only short-circuit if confirmed ----------
                vm_used, vm_value = build_blended_values(pt, all_multiples, full_text)
                nearby_text = full_text[max(0, pt['start'] - 600): min(len(full_text), pt['end'] + 600)]
                blend_confirmed = valuation_info.get('Blended') or bool(BLEND_TERMS.search(nearby_text))
                if vm_used and blend_confirmed:
                    base = {
                        'File': uploaded.name,
                        'Broker': broker,
                        'Price Target': pt.get('value'),
                        'Currency': pt.get('currency'),
                        'Valuation Multiple Used': None,
                        'Valuation Multiple Value': None,
                        'Weights': None,
                        'Blended': None,
                        'Confidence': None,
                        'Year': year_label_for_pt,
                        'Context': pt.get('context'),
                        'File Date': report_date.date() if report_date else None,
                        'DCF_Present': valuation_info.get('DCF_Present', False)
                    }
                    append_split_records(all_records, base, vm_used, vm_value, valuation_info.get('Weights'), True, 'High', dcf_params)
                    # skip normal candidate selection after emitting split rows
                    continue
                # ---------- END BLEND-FIRST SHORTCIRCUIT ----------

                dcf_near = bool(re.search(r'(?i)\b(DCF|discounted cash flow|DCF-derived|DCF derived)\b',
                                          full_text[max(0, pt['start'] - 400): min(len(full_text), pt['end'] + 400)]))

                def candidates_in_source(src, within_window=True):
                    cand = []
                    for m in all_multiples:
                        if m['source'] != src:
                            continue
                        if within_window and not (m['start'] >= win_start and m['end'] <= win_end):
                            continue
                        mult_mid = (m['start'] + m['end']) / 2
                        dist = abs(pt_mid - mult_mid)
                        mscore = 0.0
                        if m.get('canonical'):
                            if 'EV/FCF' in m['canonical']:
                                mscore += 12.0
                            elif 'EV/EBITDA' in m['canonical']:
                                mscore += 9.0
                            elif 'EV/Sales' in m['canonical']:
                                mscore += 8.0
                            elif 'P/E' in m['canonical']:
                                mscore += 4.0
                            elif 'DCF' in m['canonical']:
                                mscore += 10.0
                        if m.get('canonical') and 'EV/FCF' in m['canonical']:
                            if 8.0 <= m['value'] <= 60.0:
                                mscore += 6.0
                            elif m['value'] < 6.0:
                                mscore -= 6.0
                        ctx_window = full_text[max(0, m['start'] - 120): m['end'] + 120]
                        if re.search(r'(?i)\b(based on|derived from|using|implies|which implies|our PT)\b', ctx_window):
                            mscore += 7.0
                        if m['source'] == 'valuation_pages':
                            mscore += 1.0
                        mscore -= (dist / 1000.0)
                        cand.append({**m, 'distance': dist, 'match_score': round(mscore, 3)})
                    cand.sort(key=lambda x: (-x['match_score'], x['distance']))
                    return cand

                val_in_window = candidates_in_source('valuation_pages', within_window=True)
                val_any = candidates_in_source('valuation_pages', within_window=False)
                ft_in_window = candidates_in_source('full_text', within_window=True)
                ft_any = candidates_in_source('full_text', within_window=False)

                chosen = next((lst[0] for lst in [val_in_window, val_any, ft_in_window, ft_any] if lst), None)

                if chosen:
                    blend_ctx = full_text[max(0, pt['start'] - 400): min(len(full_text), pt['end'] + 400)]
                    if BLEND_TERMS.search(blend_ctx):
                        methods_in_blend = set()
                        if re.search(r'(?i)EV\s*\/?\s*S|EV\s*\/?\s*REVENUE|EV\s*\/?\s*SALES', blend_ctx):
                            methods_in_blend.add('EV/Sales')
                        if re.search(r'(?i)EV\s*\/?\s*FCF|EV[\s\/\-\_\w]{0,12}FCF', blend_ctx):
                            methods_in_blend.add('EV/FCF')
                        if re.search(r'(?i)EV\s*\/?\s*EBITDA', blend_ctx):
                            methods_in_blend.add('EV/EBITDA')
                        # Default preference for blend primary method
                        BLEND_PRIMARY_PREFERENCE = 'EV/FCF'
                        if 'EV/FCF' in methods_in_blend and 'EV/Sales' in methods_in_blend and BLEND_PRIMARY_PREFERENCE == 'EV/FCF':
                            def find_in_list(lst, canon):
                                for it in lst:
                                    if it.get('canonical') and canon in it.get('canonical'):
                                        return it
                                return None
                            prefer = None
                            for src_list in (val_in_window or [], val_any or [], ft_in_window or [], ft_any or []):
                                prefer = find_in_list(src_list, 'EV/FCF')
                                if prefer:
                                    chosen = prefer
                                    break

                # fallback to chosen result
                local_label = chosen.get('canonical') or chosen.get('label') if chosen else None
                if not local_label and chosen:
                    ctx_window = full_text[max(0, chosen['start'] - 160): min(len(full_text), chosen['end'] + 160)]
                    local_label = detect_canonical_from_context(ctx_window) or local_label
                local_value = f"{chosen.get('value')}x" if chosen and chosen.get('value') is not None else None

                # DCF override: append DCF if present
                methods_combined = []
                if dcf_near:
                    if local_label:
                        methods_combined = [local_label, 'DCF'] if local_label != 'DCF' else ['DCF']
                    else:
                        methods_combined = ['DCF']
                else:
                    if local_label:
                        methods_combined = [local_label]

                if methods_combined:
                    valuation_multiple_used = ' / '.join(m for m in methods_combined if m)
                else:
                    valuation_multiple_used = None

                # final blended safety check and doc-level fallback
                nearby_text = full_text[max(0, pt['start'] - 600): min(len(full_text), pt['end'] + 600)]
                if valuation_multiple_used and ('EV/Sales' not in valuation_multiple_used or 'EV/FCF' not in valuation_multiple_used):
                    if re.search(r'(?i)EV\s*/?\s*(Sales|Revenue)', nearby_text) and re.search(r'(?i)EV\s*/?\s*FCF', nearby_text):
                        vmu2, vmv2 = build_blended_values(pt, all_multiples, full_text)
                        # only accept the blended override if the doc/page/nearby confirmed blend
                        if vmu2 and (valuation_info.get('Blended') or bool(BLEND_TERMS.search(nearby_text))):
                            valuation_multiple_used = vmu2
                            local_value = vmv2

                year_label = year_label_for_pt
                if chosen:
                    year_label = chosen.get('year_label', year_label_for_pt)

                base = {
                    'File': uploaded.name,
                    'Broker': broker,
                    'Price Target': pt.get('value'),
                    'Currency': pt.get('currency'),
                    'Valuation Multiple Used': None,
                    'Valuation Multiple Value': None,
                    'Weights': None,
                    'Blended': None,
                    'Confidence': None,
                    'Year': year_label,
                    'Context': pt.get('context'),
                    'File Date': report_date.date() if report_date else None,
                    'DCF_Present': valuation_info.get('DCF_Present', False)
                }

                append_split_records(all_records, base, valuation_multiple_used or local_label, local_value, valuation_info.get('Weights'), valuation_info.get('Blended'), 'High' if local_value or valuation_multiple_used else 'Low', dcf_params)

        # show per-file raw & summary
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.subheader(f"Raw Snippet — {uploaded.name}")
            st.text_area("Raw (short)", value=normalize(full_text)[:1800], height=260)
            with st.expander("Show more raw text"):
                st.text(normalize(full_text)[:20000])
        with col_r:
            st.subheader("Auto-detected Summary")
            st.write(f"- Broker: **{broker}**")
            st.write(f"- Report date: **{report_date.date() if report_date else 'N/A'}**")
            st.write(f"- Valuation methods detected: **{valuation_info.get('Valuation_Methods') or 'None'}**")
            st.write(f"- Blended: **{valuation_info.get('Blended')}**")
            st.write(f"- DCF Present: **{valuation_info.get('DCF_Present', False)}**")
            st.write(f"- Price targets found: **{len(pts)}**")

# consolidated output
if all_records:
    # Ensure File Date column exists for all rows
    for rec in all_records:
        if 'File Date' not in rec:
            rec['File Date'] = None

    df = pd.DataFrame(all_records)
    st.subheader("Auto-detected Table (consolidated)")
    show_cols = ["File","File Date","Broker","Price Target","Currency","Valuation Multiple Used","Valuation Multiple Value","Weights","Blended","Confidence","Year","DCF_Present","WACC","Terminal Growth Rate","Equity Risk Premium","Risk Free Rate","FCF Estimates","Discount Period","Enterprise Value"]
    st.dataframe(df[show_cols], use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("Download CSV", csv, "val1_fixed_split_v3_extracted_data.csv", "text/csv")

    excel_bytes = io.BytesIO()
    df.to_excel(excel_bytes, index=False, sheet_name="Extracted", engine="openpyxl")
    excel_bytes.seek(0)
    st.download_button("Download Excel", excel_bytes.read(), "val1_fixed_split_v3_extracted_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")