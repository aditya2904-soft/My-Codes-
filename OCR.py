#!/usr/bin/env python3
"""
val_extractor_ocr.py

Single-file valuation extraction tool with OCR support for scanned/image PDFs.
Outputs CSV / Excel with detected: Broker, Price Target, Multiples used, Blended flags, DCF params, Year, File date, Confidence.

Author: Generated for user (Aditya) — formal, data-oriented, production-ready.
"""

import io
import os
import re
import sys
import argparse
from datetime import datetime
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta

import pandas as pd
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# Optional: point to tesseract executable on Windows (edit if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------------
# Tunables / thresholds
# ----------------------------
MIN_TEXT_CHARS_FOR_NON_OCR = 300      # If extracted text shorter than this -> run OCR
SEARCH_RADIUS_CHARS = 600
YEAR_SEARCH_WINDOW = 1000
BLEND_PRIMARY_PREFERENCE = 'EV/FCF'
DEBUG = False

# ----------------------------
# Broker list (same as your prior)
# ----------------------------
BROKERS = [
    "Goldman Sachs","Morgan Stanley","J.P. Morgan","JP Morgan","JPMorgan","Morning Star",
    "UBS","Barclays","Bank of America","BAML","Deutsche Bank","DB",
    "Credit Suisse","Citi Research","Citi","Citigroup","Jefferies","Evercore","RBC","Craig Hallum","OPCO",
    "Wells Fargo","Nomura","Piper Sandler","Roth","D.A. Davidson",
    "Rosenblatt","Macquarie","Canaccord","Cowen","Stifel","HSBC",
    "Berenberg","Bernstein","BNP Paribas","Societe Generale","Loop Capital","Loop",
    "Mizuho","KeyBanc","Oppenheimer","SunTrust","Raymond James",
    "Telsey Advisory Group","Telsey","Guggenheim","Lazard","Morrison Foerster","B Riley","B. Riley","SIG","Seaport Global",
    "Moelis","PJT Partners","Sandler O'Neill","Sandler","BMO Capital Markets",
    "Scotiabank","TD Securities","Canaccord Genuity","Stifel Nicolaus",
    "FBN", "Wolfe Research", "BofA", "TD Cowen", "KBW", "RBC Capital Markets", "D.A. Davidson","Baird","Cantor","Melius"
]
BROKERS_SORTED = sorted(BROKERS, key=lambda x: -len(x))

# ----------------------------
# Patterns and utilities (condensed, production-ready)
# ----------------------------

PRICE_TARGET_RE = re.compile(r'''(?xi)
    (?:
        (?:
            target\s*price|price\s*target|target\s*:|price\s*objective|target\s*price(?:\s*/\s*base\s*case)?
        )
        [\s:\-–—]*([A-Z]{0,3}[\$\€\£\¥]?)?\s*([0-9]{1,5}(?:[,\.][0-9]{3})*(?:\.[0-9]+)?)
    )
    |
    (?:([\$\€\£\¥])\s*([0-9]{1,5}(?:[,\.][0-9]{3})*(?:\.[0-9]+)?)\s*(?:target\s*price|price\s*target)?)
''', re.MULTILINE | re.DOTALL)

PRICE_GOOD_CTX = re.compile(r'(?i)\b(target price|price objective|price target|valuation methodology|our target|base case)\b')
PRICE_BAD_CTX = re.compile(r'(?i)\b(current price|trading at|last close|as of|market price|price \(as of|price as of)\b')

MULTIPLE_PATTERN = re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\s*(EV\s*\/?\s*FCF|EV\s*\/?\s*EBITDA|EV\s*\/?\s*SALES|EV\s*\/?\s*REVENUE|P\s*\/?\s*E|P\s*\/?\s*S|PEG|FCF|EBITDA|SALES|REVENUE)', re.I)
LOOSE_MULT_RE = re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*[x×X]\b', re.I)

BLEND_TERMS = re.compile(r'(?i)\b(blend|blended|average|equal weight|equally[-\s]*weighted|weighted|mix|weighting|50/50|50-50|60/40)\b')

FY_RE = re.compile(r'\bFY[\s\-]?(\d{2,4})\b', re.I)
CY_RE = re.compile(r'\bCY[\s\-]?(\d{2,4})\b', re.I)
YEAR_4_RE = re.compile(r'\b(19|20)\d{2}\b')

REPORT_DATE_PATTERNS = [
    r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}',
    r'\d{4}-\d{2}-\d{2}',
    r'\d{1,2}/\d{1,2}/\d{2,4}'
]

def parse_number(s):
    if s is None: return None
    s = str(s).strip().replace(',', '')
    s = re.sub(r'^[^\d\-]+', '', s)
    try:
        return float(s)
    except Exception:
        return None

def normalize(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()

# ----------------------------
# OCR-aware text extraction
# ----------------------------
def extract_text_with_ocr(file_bytes, dpi=300, ocr_lang='eng'):
    """
    Returns (full_text, pages_text_list)
    Strategy:
      - Try pdfplumber text extraction.
      - If text length small or obviously scanned, run OCR on each page image via pdf2image -> pytesseract.
      - Combine results and return.
    """
    pages_text = []
    full_text = ""
    # 1) Attempt native extraction with pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                txt = p.extract_text() or ""
                pages_text.append(txt)
            full_text = "\n\n".join(pages_text)
    except Exception as e:
        # if pdfplumber fails, proceed to OCR
        pages_text = []
        full_text = ""

    # 2) Decide whether to use OCR
    if len(full_text) < MIN_TEXT_CHARS_FOR_NON_OCR:
        # perform OCR on page images
        try:
            images = convert_from_bytes(file_bytes, dpi=dpi)
            pages_text = []
            for i, img in enumerate(images):
                # ensure image mode OK
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # pytesseract OCR
                text = pytesseract.image_to_string(img, lang=ocr_lang)
                pages_text.append(text)
            full_text = "\n\n".join(pages_text)
        except Exception as ex:
            # last resort (pdfplumber may have failed earlier) - return what we have
            if DEBUG:
                print("OCR failed:", ex)
            # ensure pages_text not empty
            if not pages_text:
                pages_text = [full_text]
    return full_text, pages_text

# ----------------------------
# Detection & inference helpers
# ----------------------------
def detect_broker(full_text, pages):
    txt = (full_text or "").lower()
    first_page = pages[0] if pages else full_text or ""
    first_block = (first_page or "")[:1200].lower()
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
        matches = sum(1 for t in tokens if t in tokens_text)
        if matches >= min(2, len(tokens)):
            return b
    lines = [ln.strip() for ln in (first_page or "").splitlines() if ln.strip()]
    return lines[0] if lines else "Unknown"

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
    if report_date:
        return f"CY {report_date.year}"
    m = YEAR_4_RE.search(window)
    if m:
        return f"CY {int(m.group(0))}"
    return f"CY {datetime.now().year}"

# ----------------------------
# Price target extraction (concise, robust)
# ----------------------------
def extract_price_targets(text):
    pts = []
    # quick anchor search first
    for m in PRICE_TARGET_RE.finditer(text):
        groups = m.groups()
        # groups: (maybe currency prefix, number) or (symbol, number)
        currency = ""
        val = None
        for g in groups:
            if not g: continue
            if re.search(r'[\$\€\£\¥]|USD|US\$', str(g), re.I):
                currency = str(g).strip()
            if re.search(r'\d', str(g)):
                vv = parse_number(g)
                if vv is not None:
                    val = vv
        if val is not None and 1 <= val <= 50000:
            pts.append({'value': val, 'currency': currency, 'start': m.start(), 'end': m.end(), 'context': text[max(0,m.start()-200): m.end()+200]})
    # dedupe and score: prefer contexts containing 'target'
    if not pts:
        return []
    # sort by simple heuristics
    pts.sort(key=lambda x: (-('target' in (x.get('context') or '').lower()), -x['value']))
    return pts

# ----------------------------
# Multiples extraction (similar to earlier)
# ----------------------------
def extract_all_multiples(full_text, pages, report_date):
    # create page offsets
    page_starts = []
    running = 0
    sep = "\n\n"
    for p in pages:
        page_starts.append(running)
        running += len(p or "") + len(sep)

    multiples = []
    # search in entire text
    for m in MULTIPLE_PATTERN.finditer(full_text):
        s = m.start(); e = m.end()
        num = parse_number(m.group(1)); label = (m.group(2) or "").strip()
        if num is None or not (1.5 <= num <= 200): continue
        canonical = None
        label_up = label.upper()
        if 'FCF' in label_up and 'EV' in label_up:
            canonical = 'EV/FCF'
        elif 'EBITDA' in label_up and 'EV' in label_up:
            canonical = 'EV/EBITDA'
        elif re.search(r'EV\s*\/\s*(SALES|REVENUE)', label_up):
            canonical = 'EV/Sales'
        elif re.search(r'P\s*\/\s*E', label_up):
            canonical = 'P/E'
        ctx = normalize(full_text[max(0, m.start()-80): m.end()+80])
        item = {'start': s, 'end': e}
        year_label = infer_year_label(item, full_text, report_date)
        multiples.append({'value': num, 'label': label if label else None, 'canonical': canonical, 'start': s, 'end': e, 'context': ctx, 'year_label': year_label})

    # loose matches
    for m in LOOSE_MULT_RE.finditer(full_text):
        s = m.start(); e = m.end()
        num = parse_number(m.group(1))
        if num is None or not (1.5 <= num <= 200): continue
        ctx_near = full_text[max(0, m.start()-160): m.end()+160]
        # attempt to infer
        canonical = None
        if re.search(r'(?i)EV[\s\/\-\_\w]{0,20}FCF', ctx_near):
            canonical = 'EV/FCF'
        if re.search(r'(?i)EV[\s\/\-\_\w]{0,20}EBITDA', ctx_near):
            canonical = 'EV/EBITDA'
        label = canonical.replace('/', '') if canonical else None
        item = {'start': s, 'end': e}
        year_label = infer_year_label(item, full_text, report_date)
        multiples.append({'value': num, 'label': label, 'canonical': canonical, 'start': s, 'end': e, 'context': normalize(ctx_near), 'year_label': year_label})

    # dedupe by position+value
    dedup = []
    seen = set()
    for m in multiples:
        key = (round(m['value'], 4), m['start'])
        if key in seen: continue
        seen.add(key); dedup.append(m)
    dedup.sort(key=lambda x: x['start'])
    return dedup

# ----------------------------
# Valuation method detection & DCF param extraction (concise)
# ----------------------------
VAL_PATTERNS = {
    r'\bDCF\b': 'DCF',
    r'discounted\s+cash\s+flow': 'DCF',
    r'\bWACC\b': 'WACC',
    r'EV\s*\/\s*FCF': 'EV/FCF',
    r'EV\s*\/\s*EBITDA': 'EV/EBITDA',
    r'EV\s*\/\s*SALES': 'EV/Sales',
    r'P\s*\/\s*E': 'P/E',
    r'FCF\s*yield': 'FCF Yield'
}
VAL_PAT_TUPLES = [(re.compile(k, re.I), v) for k, v in VAL_PATTERNS.items()]

def extract_dcf_parameters(valuation_text):
    dcf_params = {}
    wacc_pat = re.compile(r'(?i)(?:wacc|weighted\s+average\s+cost\s+of\s+capital|cost\s+of\s+capital|coc)[^\d]{0,20}(\d+(?:\.\d+)?)\s*%?')
    m = wacc_pat.search(valuation_text)
    dcf_params['WACC'] = m.group(1) if m else None

    tgr_pat = re.compile(r'(?i)terminal\s*growth\s*rate[^0-9]{0,10}(\d+(?:\.\d+)?)\s*%?')
    m = tgr_pat.search(valuation_text)
    dcf_params['Terminal Growth Rate'] = m.group(1) if m else None

    erp_pat = re.compile(r'(?i)(?:equity\s+)?risk\s+premium[^0-9]{0,10}(\d+(?:\.\d+)?)\s*%?')
    m = erp_pat.search(valuation_text)
    dcf_params['Equity Risk Premium'] = m.group(1) if m else None

    rfr_pat = re.compile(r'(?i)risk[- ]?free\s+rate[^0-9]{0,10}(\d+(?:\.\d+)?)\s*%?')
    m = rfr_pat.search(valuation_text)
    dcf_params['Risk Free Rate'] = m.group(1) if m else None

    return dcf_params

def extract_valuation_methods(full_text, pages):
    valuation_text = ""
    for p in pages:
        if re.search(r'(?i)\b(valuation|valuation methodology|price target calculation|investment thesis|valuation and risks|price objective)\b', p or ""):
            valuation_text += p + "\n\n"
    if not valuation_text:
        valuation_text = full_text or ""

    found = set()
    for patt, canon in VAL_PAT_TUPLES:
        if patt.search(valuation_text):
            found.add(canon)
    # fuzzy checks
    if re.search(r'(?i)EV[\s\/\-\_\w]{0,20}FCF', valuation_text):
        found.add('EV/FCF')
    if re.search(r'(?i)EV[\s\/\-\_\w]{0,20}EBITDA', valuation_text):
        found.add('EV/EBITDA')
    blended = bool(BLEND_TERMS.search(valuation_text))
    dcf_params = extract_dcf_parameters(valuation_text)
    return {
        'Valuation_Methods': '; '.join(sorted(found)) if found else None,
        'Blended': blended,
        'DCF_Present': 'DCF' in found,
        **dcf_params
    }

# ----------------------------
# Assemble rows & output
# ----------------------------
def append_record(records, base, valuation_used, valuation_value, weights, blended_flag, confidence, dcf_params):
    r = base.copy()
    r.update({
        'Valuation Multiple Used': valuation_used,
        'Valuation Multiple Value': valuation_value,
        'Weights': weights,
        'Blended': blended_flag,
        'Confidence': confidence
    })
    r.update(dcf_params or {})
    records.append(r)

# ----------------------------
# Main per-file processing
# ----------------------------
def process_file(file_path, ocr_lang='eng'):
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    full_text, pages = extract_text_with_ocr(file_bytes, ocr_lang=ocr_lang)

    # report date detection from first page heuristics
    report_date = None
    first_page = pages[0] if pages else full_text or ''
    for pat in REPORT_DATE_PATTERNS:
        m = re.search(pat, first_page, re.I)
        if m:
            try:
                report_date = dateparser.parse(m.group(0)); break
            except Exception:
                pass

    broker = detect_broker(full_text, pages)
    pts = extract_price_targets(full_text)
    valuation_info = extract_valuation_methods(full_text, pages)
    multiples = extract_all_multiples(full_text, pages, report_date)
    dcf_params = {k: valuation_info.get(k, None) for k in ['WACC', 'Terminal Growth Rate', 'Equity Risk Premium', 'Risk Free Rate']}

    records = []
    # if no PT found -> doc-level pseudo-pt and try to infer blended multiples
    if not pts:
        doc_pt = {'start': 0, 'end': len(full_text or "")}
        year_label = infer_year_label(doc_pt, full_text, report_date)
        # derive a simple vm used/value: prefer EV/FCF if present among multiples, else first canonical
        vm_used = None; vm_value = None
        for m in multiples:
            if m.get('canonical') == 'EV/FCF':
                vm_used = 'EV/FCF'; vm_value = f"{m['value']}x"; break
        if not vm_used and multiples:
            vm_used = multiples[0].get('canonical') or multiples[0].get('label')
            vm_value = f"{multiples[0]['value']}x"
        base = {
            'File': os.path.basename(file_path),
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
        append_record(records, base, vm_used, vm_value, None, valuation_info.get('Blended'), 'Low' if not vm_used else 'Medium', dcf_params)
        return records, full_text, pages

    # if PTs found: pick each and try to associate multiple
    for pt in pts:
        year_label_for_pt = infer_year_label(pt, full_text, report_date)
        pt_mid = (pt['start'] + pt['end']) / 2
        win_start = max(0, pt['start'] - SEARCH_RADIUS_CHARS)
        win_end = min(len(full_text), pt['end'] + SEARCH_RADIUS_CHARS)
        # find candidate multiples in window
        candidates = [m for m in multiples if (m['start'] >= win_start and m['end'] <= win_end)]
        chosen = None
        if candidates:
            # score candidates by canonical preference and proximity
            def score_mult(m):
                s = 0
                if m.get('canonical') == 'EV/FCF': s += 12
                if m.get('canonical') == 'EV/EBITDA': s += 9
                if m.get('canonical') == 'EV/Sales': s += 8
                dist = abs(pt_mid - ((m['start'] + m['end'])/2))
                s -= dist / 1000.0
                return s
            candidates.sort(key=lambda x: -score_mult(x))
            chosen = candidates[0]
        else:
            # fallback: any multiple in doc
            if multiples:
                chosen = multiples[0]
        local_label = chosen.get('canonical') if chosen else None
        local_value = f"{chosen.get('value')}x" if chosen and chosen.get('value') is not None else None
        # if both EV/Sales and EV/FCF exist nearby and blend confirmed, join them
        nearby_text = full_text[max(0, pt['start'] - 600): min(len(full_text), pt['end'] + 600)]
        blended_flag = valuation_info.get('Blended') or bool(BLEND_TERMS.search(nearby_text))
        if blended_flag:
            # try to find both canonical types
            found_map = {}
            for m in (candidates or multiples):
                if m.get('canonical') in ('EV/Sales', 'EV/FCF', 'EV/EBITDA'):
                    found_map[m['canonical']] = f"{m['value']}x"
            if 'EV/Sales' in found_map and 'EV/FCF' in found_map:
                vm_used = "EV/Sales + EV/FCF"
                vm_value = f"{found_map['EV/Sales']} + {found_map['EV/FCF']}"
            else:
                vm_used = local_label
                vm_value = local_value
        else:
            vm_used = local_label
            vm_value = local_value

        base = {
            'File': os.path.basename(file_path),
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
        conf = 'High' if vm_value else 'Low'
        append_record(records, base, vm_used, vm_value, None, blended_flag, conf, dcf_params)

    return records, full_text, pages

# ----------------------------
# CLI / Batch runner
# ----------------------------
def main(args):
    files = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            if fname.lower().endswith('.pdf'):
                files.append(os.path.join(args.input, fname))
    elif os.path.isfile(args.input):
        files = [args.input]
    else:
        print("Input path not found:", args.input); return

    all_records = []
    for f in files:
        if args.verbose:
            print(f"[{datetime.now().isoformat()}] Processing {f} ...")
        recs, full_text, pages = process_file(f, ocr_lang=args.ocr_lang)
        all_records.extend(recs)
        if args.debug and full_text:
            dump_txt = f"{os.path.splitext(os.path.basename(f))[0]}_extracted.txt"
            with open(dump_txt, 'w', encoding='utf-8') as fh:
                fh.write(full_text)
            if args.verbose:
                print("Wrote debug text to:", dump_txt)

    if not all_records:
        print("No records extracted.")
        return

    df = pd.DataFrame(all_records)
    out_csv = args.output or "valuation_extracted.csv"
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print("Wrote CSV:", out_csv)

    if args.xlsx:
        out_xlsx = os.path.splitext(out_csv)[0] + ".xlsx"
        df.to_excel(out_xlsx, index=False, engine="openpyxl")
        print("Wrote Excel:", out_xlsx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuation extractor with OCR support (for scanned PDF).")
    parser.add_argument("input", help="PDF file path or directory containing PDFs")
    parser.add_argument("--output", help="CSV output path (default: valuation_extracted.csv)", default="valuation_extracted.csv")
    parser.add_argument("--ocr-lang", help="Tesseract OCR language code (default: eng)", default="eng")
    parser.add_argument("--xlsx", action="store_true", help="Also write an Excel (.xlsx) output")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--debug", action="store_true", help="Write extracted raw text files for debugging")
    args = parser.parse_args()
    main(args)
