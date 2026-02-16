import re
import io
from datetime import datetime
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import pdfplumber
# ---------------------------
# Page Configuration & Styling
# ---------------------------
st.set_page_config(
    page_title="ValuationIQ Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Teal Blue Theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #008080;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #008080 0%, #20B2AA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,128,128,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #008080;
        border-bottom: 3px solid #20B2AA;
        padding-bottom: 0.8rem;
        margin-top: 2.5rem;
        font-weight: 600;
    }
    .success-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border: 2px solid #008080;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,128,128,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
        border: 2px solid #00796b;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,121,107,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #008080 0%, #00796b 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,128,128,0.2);
        border: 1px solid #20B2AA;
    }
    .upload-section {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 30px;
        border-radius: 15px;
        border: 3px dashed #008080;
        text-align: center;
        margin: 20px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #008080 0%, #00796b 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00796b 0%, #00695c 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,128,128,0.3);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8fdff 0%, #f0f8ff 100%);
    }
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #008080;
        box-shadow: 0 4px 6px rgba(0,128,128,0.1);
    }
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,128,128,0.1);
    }
    .debug-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)       
# ---------------------------
# Tunables
# ---------------------------
DEBUG = True
SEARCH_RADIUS_CHARS = 600
YEAR_SEARCH_WINDOW = 1000  # Increased for better year capture

# ---------------------------
# Broker list
# ---------------------------
BROKERS = [
    "Goldman Sachs","Morgan Stanley","J.P. Morgan","JP Morgan","JPMorgan","Morning Star",
    "UBS","Barclays","Bank of America","BAML","Deutsche Bank","DB",
    "Credit Suisse","Citi Research","Citi","Citigroup","Jefferies","Evercore","RBC","Craig Hallum","OpCO",
    "Wells Fargo","Nomura","Piper Sandler","Roth","D.A. Davidson",
    "Rosenblatt","Macquarie","Canaccord","Cowen","Stifel","HSBC",
    "Berenberg","Bernstein","BNP Paribas","Societe Generale","Loop Capital","Loop","Capital One", "Citizens", "Needham", "Redburn"
    "Mizuho","KeyBanc","Oppenheimer","SunTrust","Raymond James",
    "Telsey Advisory Group","Telsey","Guggenheim","Lazard","Morrison Foerster","B Riley","B. Riley","SIG","Seaport Global",
    "Moelis","PJT Partners","Sandler O'Neill","Sandler","BMO Capital Markets","Stephans",
    "Scotiabank","TD Securities","Canaccord Genuity","Stifel Nicolaus","Cantor","Susquehenna","NewStreet","Rosenblatt Securities",
    "FBN", "Wolfe Research", "BofA", "TD Cowen", "KBW", "RBC Capital Markets", "D.A. Davidson","Baird","Cantor","Melius","FBN Securities","Wolfe Research","Wolfe"
]
BROKERS_SORTED = sorted(BROKERS, key=lambda x: -len(x))
import re

BROKER_PATTERN = re.compile(
    r"\b("
    r"Goldman\s+Sachs|Morgan\s+Stanley|J\.?P\.?\s*Morgan|JPMorgan|Bank\s+of\s+America|BofA|BAML|"
    r"Citigroup|Citi(\s+Research)?|Barclays|UBS|Credit\s+Suisse|Deutsche\s+Bank|DB|BNP\s*Paribas|"
    r"Societe\s+Generale|HSBC|RBC(\s+Capital\s+Markets)?|Scotiabank|Macquarie|Nomura|Mizuho|"
    r"SMBC\s*Nikko|Daiwa|MUFG|Jefferies|Evercore(\s+ISI)?|BMO(\s+Capital\s+Markets)?|"
    r"Piper(\s+Sandler|Jaffray)?|Raymond\s+James|Stifel(\s+Nicolaus)?|Cowen|TD\s*Cowen|"
    r"TD\s*Securities|Oppenheimer|KeyBanc(\s+Capital\s+Markets)?|Needham|"
    r"Canaccord(\s+Genuity)?|William\s+Blair|Baird|Loop(\s+Capital)?|BTIG|Roth(\s+MKM)?|"
    r"Craig\s+Hallum|Redburn|Sanford(\s+C\.?\s*)?Bernstein|Berenberg|Lazard|Wells\s+Fargo|"
    r"OpCO|Oppenheimer\s*&\s*Co|Telsey(\s+Advisory\s+Group)?|Seaport(\s+Global)?|"
    r"Truist(\s+Securities)?|SunTrust(\s+Robinson\s+Humphrey)?|Stephens|"
    r"D\.?A\.?\s*Davidson|RayJay|Regions(\s+Securities)?|Wedbush(\s+Securities)?|"
    r"Compass\s*Point(\s+Research)?|Keefe(\s+Bruyette\s*&\s+Woods)?|KBW|"
    r"Rosenblatt(\s+Securities)?|Melius(\s+Research)?|MoffettNathanson|"
    r"Guggenheim|Morningstar|B\.?\s*Riley|FBN(\s+Securities)?|Wolfe(\s+Research)?|"
    r"Moelis(\s*&\s*Company)?|PJT(\s+Partners)?|Northland(\s+Capital)?|"
    r"Maxim(\s+Group)?|Benchmark(\s+Company)?|Sidoti(\s+&\s*Co)?|JMP(\s+Securities)?|"
    r"H\.?\s*C\.?\s*Wainwright|EF\s*Hutton|Alliance\s*Global|Barrington(\s+Research)?|"
    r"Lake\s*Street|Imperial(\s+Capital)?|Colliers(\s+Securities)?|"
    r"Northcoast(\s+Research)?|Susquehanna(\s+Financial(\s+Group)?)?|SIG|"
    r"Tudor(\s+Pickering\s+Holt)?|Loop|Morrison(\s+Foerster)?|"
    r"DA\s*Davidson|Evercore|BMO|Capital\s*One|Citizens|Truist|Regions|Penserra|"
    r"CLSA|Exane(\s+BNP\s*Paribas)?|Kepler(\s+Cheuvreux)?|Liberum(\s+Capital)?|"
    r"Peel\s*Hunt|Investec|Goodbody|Numis|Winterflood|Shore\s*Capital|Davy(\s+Research)?"
    r")\b",
    re.IGNORECASE
)
 
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
# DCF Parameters Patterns (Improved)
# ---------------------------
def extract_dcf_parameters(valuation_text):
    dcf_params = {}
    
    # WACC - bidirectional
    wacc_pat = re.compile(r'(?i)(\d+(?:\.\d+)?)\s*%?\s*(?:wacc|weighted\s+average\s+cost\s+of\s+capital)|(?:wacc|weighted\s+average\s+cost\s+of\s+capital).*?(\d+(?:\.\d+)?)\s*%?', re.I | re.S)
    wacc_match = wacc_pat.search(valuation_text)
    if wacc_match:
        dcf_params['WACC'] = wacc_match.group(1) or wacc_match.group(2)
    else:
        dcf_params['WACC'] = None
    
    # Terminal Growth Rate
    tgr_pat = re.compile(r'(?i)terminal\s*growth\s*rate.*?(\d+(?:\.\d+)?)\s*%?', re.I | re.S)
    tgr_match = tgr_pat.search(valuation_text)
    dcf_params['Terminal Growth Rate'] = tgr_match.group(1) if tgr_match else None
    
    # Equity Risk Premium
    erp_pat = re.compile(r'(?i)(?:equity\s+)?risk\s+premium.*?(\d+(?:\.\d+)?)\s*%?', re.I | re.S)
    erp_match = erp_pat.search(valuation_text)
    dcf_params['Equity Risk Premium'] = erp_match.group(1) if erp_match else None
    
    # Risk Free Rate - handle terminal and hyphen
    rfr_pat = re.compile(r'(?i)(?:terminal\s+)?risk[- ]?free\s+(?:rate).*?of\s+(\d+(?:\.\d+)?)\s*%?', re.I | re.S)
    rfr_match = rfr_pat.search(valuation_text)
    dcf_params['Risk Free Rate'] = rfr_match.group(1) if rfr_match else None
    
    # FCF Estimates - handle ranges like FY26/27 $4,039m/$4,656m
    fcf_pat = re.compile(r'(?i)FY(\d{2}(?:/\d{2})?)\s*(?:FCF\s+)?estimates?\s+to\s+\$?([,\d,]+\.?\d*(?:m)?)/\$?([,\d,]+\.?\d*(?:m)?)', re.I | re.S)
    fcf_matches = fcf_pat.findall(valuation_text)
    if fcf_matches:
        fcf_str = '; '.join([f"{fy}: ${val1}/{val2}" for fy, val1, val2 in fcf_matches])
        dcf_params['FCF Estimates'] = fcf_str
    else:
        # Fallback for single
        single_fcf_pat = re.compile(r'(?i)FY(\d{2})\s*(?:FCF\s+)?estimates?\s+to\s+\$?([,\d,]+\.?\d*(?:m)?)', re.I | re.S)
        single_matches = single_fcf_pat.findall(valuation_text)
        if single_matches:
            dcf_params['FCF Estimates'] = '; '.join([f"FY{fy}: ${val}" for fy, val in single_matches])
        else:
            dcf_params['FCF Estimates'] = None
    
    # Discount Period
    disc_pat = re.compile(r'(?i)(\d+(?:-year)?)\s*(?:modified\s+)?DCF', re.I | re.S)
    disc_match = disc_pat.search(valuation_text)
    dcf_params['Discount Period'] = disc_match.group(1) if disc_match else None
    
    # Enterprise Value
    ev_pat = re.compile(r'(?i)enterprise\s+value.*?\$?([,\d,]+\.?\d*(?:m)?)', re.I | re.S)
    ev_match = ev_pat.search(valuation_text)
    dcf_params['Enterprise Value'] = ev_match.group(1) if ev_match else None
    
    return dcf_params

# ---------------------------
# Regex patterns
# ---------------------------
BLEND_PRIMARY_PREFERENCE = 'EV/FCF'

PRICE_TARGET_RE = re.compile(r'''(?xi)
    (?:
        (?:
            12\s*m(?:o|onth)?\s*(?:price\s*target|target\s*price)|
            price\s*target|target\s*price|target\s*:|target\s*price(?:\s*/\s*base\s*case)?|
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
    r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\s*(EV\s*\/?\s*FCF|EV\s*\/?\s*EBITDA|EV\s*\/?\s*SALES|EV\s*\/?\s*REVENUE|P\s*\/?\s*E|P\s*\/?\s*S|P\s*\/?\s*B|PBV|PEG|FCF|EBITDA|SALES|REVENUE)',
    re.I
)
LOOSE_MULT_RE = re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\b', re.I)

FUZZY_EV_FCF = re.compile(r'EV[\s\/\-\_\w]{0,20}FCF', re.I)
FUZZY_EV_SALES = re.compile(r'EV[\s\/\-\_\w]{0,20}SALES|EV[\s\/\-\_\w]{0,20}REVENUE|EV\s*\/\s*S\b', re.I)
FUZZY_EV_EBITDA = re.compile(r'EV[\s\/\-\_\w]{0,20}EBITDA', re.I)

RATIO_LIKE = re.compile(r'\b\d{1,3}\s*[x×X]\b|\b\d{1,3}\s*%\b|\b\d{1,3}\s*\/\s*\d{1,3}\b', re.I)
VALUATION_KEYWORDS_RE = re.compile(r'(?i)\b(EV|FCF|EBITDA|REVENUE|SALES|P/E|P/BV|PBV|DCF|WACC|FCF yield)\b')
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

PRICE_GOOD_CTX = re.compile(r'(?i)\b(target price|price objective|price target|valuation methodology|target:|our target|base case)\b')
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
# Ticker Name Detection (FIXED)
# ---------------------------
def extract_ticker_from_filename(filename):
    """
    Extracts a potential stock ticker from the filename.
    Assumes ticker is a 2-5 alphanumeric uppercase sequence, often at the start
    or before a hyphen/space.
    """
    # Remove extension and path
    base_name = re.sub(r'\.\w+$', '', filename)
    base_name = base_name.upper()  # Convert to uppercase for consistent searching
    
    # Pattern: 2-5 uppercase letters/digits, ideally at start or before common separators
    # More flexible pattern that handles underscores, hyphens, spaces
    ticker_pat = re.compile(r'^([A-Z0-9]{2,5})[\s\_\-]|[\s\_\-]([A-Z0-9]{2,5})[\s\_\-]')
    
    # 1. Highest Priority: Ticker at the very beginning of filename
    m_start = re.match(r'^([A-Z0-9]{2,5})[\s\_\-]', base_name)
    if m_start:
        candidate = m_start.group(1)
        # Prefer tickers with letters (not pure numbers)
        if not candidate.isdigit():
            return candidate
    
    # 2. Look for ticker-like patterns elsewhere in filename
    m = ticker_pat.search(base_name)
    if m:
        # Get the non-None group
        candidate = m.group(1) or m.group(2)
        if candidate and not candidate.isdigit():  # Avoid pure numbers like dates
            return candidate
    
    # 3. Fallback: First alphanumeric sequence of 2-5 chars
    words = re.findall(r'[A-Z0-9]{2,5}', base_name)
    for word in words:
        if not word.isdigit():  # Skip pure numbers
            return word
    
    # 4. Last resort: first word if alphanumeric
    first_word = base_name.split()[0] if base_name.split() else ""
    if 2 <= len(first_word) <= 5 and first_word.isalnum():
        return first_word
        
    return "UNKNOWN"

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
# Extract DCF Parameters (Dynamic)
# ---------------------------
def extract_dcf_parameters(valuation_text):
    dcf_params = {}
    
    # WACC - bidirectional
    wacc_pat = re.compile(r'(?i)(\d+(?:\.\d+)?)\s*%?\s*(?:wacc|weighted\s+average\s+cost\s+of\s+capital)|(?:wacc|weighted\s+average\s+cost\s+of\s+capital).*?(\d+(?:\.\d+)?)\s*%?', re.I | re.S)
    wacc_match = wacc_pat.search(valuation_text)
    if wacc_match:
        dcf_params['WACC'] = wacc_match.group(1) or wacc_match.group(2)
    else:
        dcf_params['WACC'] = None
    
    # Terminal Growth Rate
    tgr_pat = re.compile(r'(?i)terminal\s*growth\s*rate.*?(\d+(?:\.\d+)?)\s*%?', re.I | re.S)
    tgr_match = tgr_pat.search(valuation_text)
    dcf_params['Terminal Growth Rate'] = tgr_match.group(1) if tgr_match else None
    
    # Equity Risk Premium
    erp_pat = re.compile(r'(?i)(?:equity\s+)?risk\s+premium.*?(\d+(?:\.\d+)?)\s*%?', re.I | re.S)
    erp_match = erp_pat.search(valuation_text)
    dcf_params['Equity Risk Premium'] = erp_match.group(1) if erp_match else None
    
    # Risk Free Rate - handle terminal and hyphen
    rfr_pat = re.compile(r'(?i)(?:terminal\s+)?risk[- ]?free\s+(?:rate).*?of\s+(\d+(?:\.\d+)?)\s*%?', re.I | re.S)
    rfr_match = rfr_pat.search(valuation_text)
    dcf_params['Risk Free Rate'] = rfr_match.group(1) if rfr_match else None
    
    # FCF Estimates - handle ranges like FY26/27 $4,039m/$4,656m
    fcf_pat = re.compile(r'(?i)FY(\d{2}(?:/\d{2})?)\s*(?:FCF\s+)?estimates?\s+to\s+\$?([,\d,]+\.?\d*(?:m)?)/\$?([,\d,]+\.?\d*(?:m)?)', re.I | re.S)
    fcf_matches = fcf_pat.findall(valuation_text)
    if fcf_matches:
        fcf_str = '; '.join([f"{fy}: ${val1}/{val2}" for fy, val1, val2 in fcf_matches])
        dcf_params['FCF Estimates'] = fcf_str
    else:
        # Fallback for single
        single_fcf_pat = re.compile(r'(?i)FY(\d{2})\s*(?:FCF\s+)?estimates?\s+to\s+\$?([,\d,]+\.?\d*(?:m)?)', re.I | re.S)
        single_matches = single_fcf_pat.findall(valuation_text)
        if single_matches:
            dcf_params['FCF Estimates'] = '; '.join([f"FY{fy}: ${val}" for fy, val in single_matches])
        else:
            dcf_params['FCF Estimates'] = None
    
    # Discount Period
    disc_pat = re.compile(r'(?i)(\d+(?:-year)?)\s*(?:modified\s+)?DCF', re.I | re.S)
    disc_match = disc_pat.search(valuation_text)
    dcf_params['Discount Period'] = disc_match.group(1) if disc_match else None
    
    # Enterprise Value
    ev_pat = re.compile(r'(?i)enterprise\s+value.*?\$?([,\d,]+\.?\d*(?:m)?)', re.I | re.S)
    ev_match = ev_pat.search(valuation_text)
    dcf_params['Enterprise Value'] = ev_match.group(1) if ev_match else None
    
    return dcf_params

# ---------------------------
# Price target extraction (FIXED)
# ---------------------------
def extract_price_targets_with_pos(text):
    pts = []
    text_norm = re.sub(r'\s+', ' ', text)

    # --- Method 1: Anchor-based search (stronger) ---
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
            
            # **FIX: Filter out low-value numbers like '12' (from 12-month) that are false positives.**
            if val is None or val > 5000:
                continue
            if val < 20: # Raised minimum from 5 to 20
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

    # --- Method 2: Keyword pattern matching (fallback) ---
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
        
        # **FIX: Filter out low-value numbers like '12' (from 12-month) that are false positives.**
        if val < 20 or val > 5000: # Raised minimum from 5 to 20
            continue
            
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
# IMPROVED: Prioritize PT-related multiples
# ---------------------------
# --------------------------------------
# IMPROVED VERSION: Prioritize PT-linked valuation multiples
# --------------------------------------
import re

def prioritize_pt_related_multiples(pt, all_multiples, full_text, window=800):
    if not all_multiples:
        return None

    pt_start = pt.get('start', 0)
    pt_end = pt.get('end', 0)
    pt_mid = (pt_start + pt_end) / 2

    win_start = max(0, pt_start - window)
    win_end = min(len(full_text), pt_end + window)

    scored_multiples = []

    for m in all_multiples:
        if not (m['start'] >= win_start and m['end'] <= win_end):
            continue

        score = 0
        mult_mid = (m['start'] + m['end']) / 2

        # Expanded local context (captures more linkage phrases)
        mult_ctx_start = max(win_start, m['start'] - 400)
        mult_ctx_end = min(win_end, m['end'] + 400)
        mult_context = full_text[mult_ctx_start:mult_ctx_end].lower()

        # ---- PRIORITY SIGNALS ----
        pt_linkage_terms = r'(derived from|based on|using|applying|implies|reflects?|target multiple|pt multiple|justifies|supports|valuation implies|assumes|indicative of|consistent with|aligned with|in line with|corresponds to|equivalent to|suggests?|indicates?|calculated at|computed at|resulting from|implied by|tied to|linked to|related to|anchored to|grounded in|follows from|stems from|comes from|from the target price)'
        if re.search(pt_linkage_terms, mult_context):
            score += 100

        # Appears in same local sentence as PT reference
        if re.search(r'\b(PT|price target|target price)\b', mult_context):
            score += 80

        # Forward-looking / forecasted signal
        if re.search(r'\b(forward|fwd|ntm|our|target|forecast|cy20\d{2})\b', mult_context):
            score += 30

        # Canonical multiple preference - FIX: Handle None case
        canonical = m.get('canonical')
        if canonical:
            canonical = canonical.upper()
            if canonical == 'EV/FCF':
                score += 25
            elif canonical == 'EV/EBITDA':
                score += 15
            elif canonical == 'EV/SALES':
                score += 10

        # ---- SPATIAL LOGIC ----
        distance = abs(pt_mid - mult_mid)
        distance_penalty = distance / 50.0
        score -= distance_penalty

        # Directional bonus: forward multiples (after PT mention) usually relevant
        if m['start'] > pt_end:
            score += 40

        # Strong penalty for current trading multiple references
        if re.search(r'(trading at|current|currently|now|as of|market price)', mult_context):
            score -= 200

        scored_multiples.append({
            'multiple': m,
            'score': score,
            'distance': distance
        })

    if not scored_multiples:
        return None

    # Sort by descending score
    scored_multiples.sort(key=lambda x: -x['score'])

    # Return the top candidate if positive score
    if scored_multiples[0]['score'] > 0:
        return scored_multiples[0]['multiple']

    return None


# ---------------------------
# ORIGINAL ROBUST: Price target extraction with strict filtering
# ---------------------------
def extract_price_targets_with_pos(text):
    pts = []
    text_norm = re.sub(r'\s+', ' ', text)

    # --- Method 1: Anchor-based search (stronger) ---
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
            
            # Filter out low-value numbers (e.g., '12' from '12-month')
            if val is None or val > 5000:
                continue
            if val < 20:  # Raised minimum from 5 to 20 to remove false positives
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

    # --- Method 2: Direct pattern matching fallback (existing PRICE_TARGET_RE) ---
    if not pts:
        candidates = []
        for m in PRICE_TARGET_RE.finditer(text_norm):
            # group logic: currency in g1 or g3, value in g2 or g4
            currency = (m.group(1) or m.group(3)) if (m.lastindex and (m.group(1) or m.group(3))) else None
            value_str = (m.group(2) or m.group(4)) if (m.lastindex and (m.group(2) or m.group(4))) else None
            if value_str:
                val = parse_number(value_str)
                if val and 10 <= val <= 5000:
                    context = text_norm[max(0, m.start()-100): min(len(text_norm), m.end()+100)]
                    if not PRICE_BAD_CTX.search(context) and not RATIO_LIKE.search(context):
                        candidates.append({
                            'currency': currency or "$",
                            'value': val,
                            'context': context,
                            'start': m.start(),
                            'end': m.end(),
                            'score': 5 + (10 if re.search(r'[\$\€\£\¥]', (m.group(1) or m.group(3) or '')) else 0)
                        })
        if candidates:
            # prefer candidate with currency and higher value (helps when both price and PT in same string)
            candidates.sort(key=lambda x: (-x['score'], -x['value'], x['start']))
            pts.append(candidates[0])

    return pts
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
def candidates_in_source(all_multiples, src, pt_mid, win_start, win_end, full_text, within_window=True):
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
            
        # NEW: Boost for linkage/PT context in fallback
        if re.search(r'(?i)\b(derived from|based on|PT|price target)\b', ctx_window):
            mscore += 10.0
            
        # NEW: Extra penalty for trading/current in fallback
        if re.search(r'(?i)\b(trading at|current|as of)\b', ctx_window):
            mscore -= 15.0
            
        if m['source'] == 'valuation_pages':
            mscore += 1.0
            
        mscore -= (dist / 1000.0)
        cand.append({**m, 'distance': dist, 'match_score': round(mscore, 3)})
        
    cand.sort(key=lambda x: (-x['match_score'], x['distance']))
    return cand
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
st.set_page_config(page_title="Phase - 1", layout="wide")
st.title("Product - Valuation Extractor - Beta")

uploaded_files = st.file_uploader("Upload one or more PDFs", type=['pdf'], accept_multiple_files=True)
all_records = []
all_full_texts = {}

if uploaded_files:
    for uploaded in uploaded_files:
        file_bytes = uploaded.read()
        full_text, pages = extract_text(file_bytes)
        all_full_texts[uploaded.name] = full_text

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
        ticker = extract_ticker_from_filename(uploaded.name) # <-- Ticker extraction using fixed logic
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

        dcf_params = {k: valuation_info.get(k, None) for k in ['WACC', 'Terminal Growth Rate', 'Equity Risk Premium', 'Risk Free Rate', 'FCF Estimates', 'Discount Period', 'Enterprise Value']}

        # ----- if no price targets found, doc-level blended detection (only split if valuation_info indicates blended)
        if not pts:
            # build doc-level pseudo-pt
            doc_pt = {'start': 0, 'end': len(full_text or "")}
            year_label = infer_year_label(doc_pt, full_text, report_date)
            vm_used, vm_value = build_blended_values(doc_pt, all_multiples, full_text)

            base = {
                'File': uploaded.name,
                'Ticker': ticker, # <-- Ticker added to base record
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
                        'Ticker': ticker, # <-- Ticker added to base record
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

                
                # NEW: Only prioritize if linkage + PT context present; else fallback to normal
                pt_related_multiple = None
                # Check if any multiple has linkage context
                for m in all_multiples:
                    ctx = full_text[max(0, m['start'] - 300): min(len(full_text), m['end'] + 300)]
                    if re.search(r'(derived from|based on|using|implies)', ctx, re.I) and re.search(r'\b(PT|price target)\b', ctx, re.I):
                        pt_related_multiple = prioritize_pt_related_multiples(pt, all_multiples, full_text)
                        break
                
                if pt_related_multiple:
                    chosen = pt_related_multiple
                    if DEBUG:
                        st.info(f"🔍 Selected PT-related multiple: {chosen.get('canonical')} {chosen.get('value')}x")
                else:
                    # Fall back to ORIGINAL candidate selection

                    val_in_window = candidates_in_source(all_multiples, 'valuation_pages', pt_mid, win_start, win_end, full_text, within_window=True)
                    val_any = candidates_in_source(all_multiples, 'valuation_pages', pt_mid, win_start, win_end, full_text, within_window=False)
                    ft_in_window = candidates_in_source(all_multiples, 'full_text', pt_mid, win_start, win_end, full_text, within_window=True)
                    ft_any = candidates_in_source(all_multiples, 'full_text', pt_mid, win_start, win_end, full_text, within_window=False)

                    chosen = next((lst[0] for lst in [val_in_window, val_any, ft_in_window, ft_any] if lst), None)

                if chosen:
                    blend_ctx = full_text[max(0, pt['start'] - 400): min(len(full_text), pt['end'] + 400)]
                    if BLEND_TERMS.search(blend_ctx):
                        methods_in_blend = set()
                        if re.search(r'(?i)EV\s*/?\s*(Sales|Revenue)', blend_ctx):
                            methods_in_blend.add('EV/Sales')
                        if re.search(r'(?i)EV\s*/?\s*FCF|EV[\s\/\-\_\w]{0,12}FCF', blend_ctx):
                            methods_in_blend.add('EV/FCF')
                        if re.search(r'(?i)EV\s*/?\s*EBITDA', blend_ctx):
                            methods_in_blend.add('EV/EBITDA')
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
                    'Ticker': ticker, # <-- Ticker added to base record
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
            st.write(f"- Ticker: **{ticker}**") # <-- Ticker shown in summary
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
    # <-- Ticker added to show_cols
    show_cols = ["File","Ticker","File Date","Broker","Price Target","Currency","Valuation Multiple Used","Valuation Multiple Value","Weights","Blended","Confidence","Year","DCF_Present","WACC","Terminal Growth Rate","Equity Risk Premium","Risk Free Rate","FCF Estimates","Discount Period","Enterprise Value"]
    st.dataframe(df[show_cols], use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("Download CSV", csv, "val1_fixed_split_v3_extracted_data.csv", "text/csv")

    excel_bytes = io.BytesIO()
    df.to_excel(excel_bytes, index=False, sheet_name="Extracted", engine="openpyxl")
    excel_bytes.seek(0)
    st.download_button("Download Excel", excel_bytes.read(), "val1_fixed_split_v3_extracted_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# Chatbot Query Responder
if all_records and all_full_texts:
    st.markdown("---")
    st.markdown('<div class="sub-header">🤖 Document Query Assistant</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about the content of the uploaded PDFs using semantic search.")
    
    query = st.text_input("Enter your question:", placeholder="e.g., What is the risk free rate used in the DCF?")
    
    if query:
        with st.spinner("Searching through documents..."):
            snippets = []
            for file, text in all_full_texts.items():
                if not text:
                    continue
                sents = re.split(r'(?<=[\.!?])\s+', text)
                for i in range(len(sents)):
                    window = sents[i]
                    if i+1 < len(sents):
                        window = (sents[i] + ' ' + sents[i+1]).strip()
                    if len(window) > 10:
                        snippets.append((file, window))

            if not snippets:
                st.warning("No text data found for query matching.")
            else:
                corpus = [s[1] for s in snippets]
                try:
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
                    tfidf_matrix = vectorizer.fit_transform(corpus)
                    query_vec = vectorizer.transform([query])
                    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
                    top_n = min(8, len(sims))
                    top_indices = sims.argsort()[-top_n:][::-1]
                    
                    st.markdown("### 🔍 Most Relevant Snippets:")
                    for idx in top_indices:
                        if sims[idx] > 0.1:
                            file, sentence = snippets[idx]
                            score = sims[idx]
                            with st.expander(f"**From {file}** (Relevance: {score:.3f})"):
                                st.write(sentence.strip())
                except Exception as e:
                    st.error(f"Semantic search failed: {str(e)}")
                    st.info("Falling back to keyword matching...")
                    q_words = re.findall(r'\w+', query.lower())
                    results = []
                    for file, text in all_full_texts.items():
                        sentences = re.split(r'(?<=[\.!?])\s+', text)
                        relevant = []
                        for sent in sentences:
                            sent_lower = sent.lower()
                            if all(word in sent_lower for word in q_words):
                                relevant.append(sent)
                        if relevant:
                            results.append(f"**From {file}:**\n" + "\n\n".join(relevant[:3]))
                    if results:
                        st.markdown("\n\n".join(results))
                    else:
                        st.warning("No relevant snippets found using keyword fallback.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #008080; padding: 20px;'>
    <p style='font-size: 1.1rem; font-weight: 600;'>Valuation • Financial Analysis Toolkit</p>
    <p style='font-size: 0.9rem; color: #668;'>Powered by Streamlit • Advanced Document Processing</p>
</div>
""", unsafe_allow_html=True)


                        