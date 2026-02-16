import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="FMP Earnings Transcript Extractor", layout="wide")

st.title("Earnings Call Transcript — FinancialModelingPrep (Local)")
st.markdown(
    """
    Select a ticker, year and quarter to fetch earnings call transcripts from FinancialModelingPrep.

    Notes:
    - You can either paste an API key below or set the environment variable `FMP_API_KEY`.
    - Results are cached to reduce repeated API calls during your session.
    """
)

# Sidebar inputs
with st.sidebar.form(key="input_form"):
    ticker = st.text_input("Ticker symbol (e.g. MNDY)", value="MNDY").strip().upper()
    year = st.number_input("Year", min_value=1990, max_value=2099, value=2024, step=1)
    quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=1)
    default_key = os.environ.get("FMP_API_KEY", "")
    api_key = st.text_input("FMP API Key", value=default_key, type="password")
    submit = st.form_submit_button("Fetch Transcript")

API_URL_TEMPLATE = (
    "https://financialmodelingprep.com/stable/earning-call-transcript"
    "?symbol={symbol}&year={year}&quarter={quarter}&apikey={apikey}"
)

@st.cache_data(ttl=60 * 60)
def fetch_transcript(symbol: str, year: int, quarter: int, apikey: str):
    """Fetch transcript data from FinancialModelingPrep API.

    Returns JSON on success or raises an exception.
    """
    if not apikey:
        raise ValueError("No API key provided. Please provide an API key in the sidebar or set FMP_API_KEY.")

    url = API_URL_TEMPLATE.format(symbol=symbol, year=year, quarter=quarter, apikey=apikey)
    resp = requests.get(url, timeout=15)

    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception as e:
            raise ValueError(f"Failed to decode JSON response: {e}")

        # The API returns an array (possibly empty) or an object depending on the endpoint
        if not data:
            return []
        return data
    elif resp.status_code == 401:
        raise PermissionError("Unauthorized: Invalid API key or access denied.")
    else:
        raise ConnectionError(f"API request failed (status {resp.status_code}): {resp.text}")


def extract_primary_items(json_payload):
    """Normalize the JSON payload into a table-friendly structure.

    The structure returned by the FMP transcript endpoint can vary. This helper
    extracts the most useful fields into a DataFrame.
    """
    # If payload is a list of items
    if isinstance(json_payload, list):
        rows = []
        for item in json_payload:
            # Common fields seen in FMP transcript responses
            rows.append({
                "symbol": item.get("symbol"),
                "date": item.get("date"),
                "title": item.get("title") or item.get("transcriptTitle") or None,
                "type": item.get("type"),
                "transcript": item.get("transcript") or item.get("content") or item.get("text"),
                "raw": item,
            })
        return pd.DataFrame(rows)

    # If payload is a dict with a single transcript
    if isinstance(json_payload, dict):
        item = json_payload
        row = {
            "symbol": item.get("symbol"),
            "date": item.get("date"),
            "title": item.get("title") or item.get("transcriptTitle") or None,
            "type": item.get("type"),
            "transcript": item.get("transcript") or item.get("content") or item.get("text"),
            "raw": item,
        }
        return pd.DataFrame([row])

    return pd.DataFrame([])



if submit:
    st.sidebar.info("Request sent — fetching data...")
    try:
        payload = fetch_transcript(ticker, int(year), int(quarter), api_key)

        if not payload:
            st.warning("No transcript returned for the selected ticker/year/quarter.")
        else:
            df = extract_primary_items(payload)

            # Summary box
            st.markdown("### Results")
            st.write(f"Found {len(df)} transcript item(s) for {ticker} — Year: {year}, Q{quarter}.")

            # Show table with metadata
            meta_cols = [c for c in df.columns if c != "raw"]
            st.dataframe(df[meta_cols].fillna("-").style.format({"date": lambda v: v}))

            # Let user pick which item to inspect
            idx = st.number_input("Select row index to view transcript", min_value=0, max_value=max(0, len(df) - 1), value=0, step=1)
            selected = df.iloc[int(idx)]

            st.subheader(f"Transcript — {selected.get('title') or ticker}")
            transcript_text = selected.get("transcript") or ""

            if not transcript_text:
                st.info("No transcript text found in the response; showing raw JSON below.")
                st.json(selected.get("raw"))
            else:
                st.text_area("Transcript", value=transcript_text, height=400)

                # Simple text analytics: word count and first/last 500 chars preview
                word_count = len(transcript_text.split())
                st.markdown(f"**Word count:** {word_count:,}")
                st.markdown("**Preview (first 500 chars):**")
                st.code(transcript_text[:500] + ("..." if len(transcript_text) > 500 else ""))

                # Download button
                file_name = f"{ticker}_{year}_Q{quarter}_transcript.txt"
                st.download_button(label="Download transcript as .txt", data=transcript_text, file_name=file_name, mime="text/plain")

            # Show raw JSON option
            with st.expander("Raw JSON response"):
                st.json(payload)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

else:
    st.info("Enter parameters in the sidebar and click 'Fetch Transcript' to begin.")

# Footer
st.markdown("---")
st.caption("Built for local testing with FinancialModelingPrep's earning-call-transcript endpoint.")
st.caption("Developed by Aditya Shivhare UBBqWIglgSTDpX41T8yoAPenzfIBmUIo")