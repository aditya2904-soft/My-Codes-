import streamlit as st
import numpy as np
from scipy.stats import norm

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="BSM Earnings Overlay", layout="wide")
st.title("📊 BSM Earnings Overlay — Implied Volatility Engine")
st.caption("Market expectations decoder for earnings events")

# ======================================================
# BLACK–SCHOLES FUNCTIONS
# ======================================================
def bsm_call_price(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bsm_put_price(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bsm_vega(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility_call(
    market_price, S, K, r, T, initial_guess=0.30, tol=1e-6, max_iter=100
):
    sigma = initial_guess
    for _ in range(max_iter):
        price = bsm_call_price(S, K, r, T, sigma)
        vega = bsm_vega(S, K, r, T, sigma)

        if vega < 1e-8:
            return np.nan

        sigma -= (price - market_price) / vega

        if abs(price - market_price) < tol:
            return sigma

    return np.nan

# ======================================================
# INPUT PANEL
# ======================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("Market Inputs")
    S = st.number_input("Spot Price (S₀)", value=100.0)
    K = st.number_input("Strike Price (ATM)", value=100.0)
    call_mkt_price = st.number_input("ATM Call Market Price", value=5.0)

with c2:
    st.subheader("Event Timing")
    days = st.number_input("Days to Earnings", value=7)
    r = st.number_input("Risk-Free Rate", value=0.05)

with c3:
    st.subheader("Fundamental View")
    upside = st.number_input("Fundamental Upside (%)", value=15.0)
    downside = st.number_input("Fundamental Downside (%)", value=10.0)

with c4:
    st.subheader("Model Controls")
    init_vol = st.number_input("IV Initial Guess", value=0.30)

# ======================================================
# CORE CALCULATIONS
# ======================================================
T = days / 365

iv = implied_volatility_call(
    market_price=call_mkt_price,
    S=S,
    K=K,
    r=r,
    T=T,
    initial_guess=init_vol
)

if not np.isnan(iv):
    call_val = bsm_call_price(S, K, r, T, iv)
    put_val = bsm_put_price(S, K, r, T, iv)
    implied_move_pct = iv * np.sqrt(T)
    implied_move_val = S * implied_move_pct
else:
    call_val = put_val = implied_move_pct = implied_move_val = np.nan

# ======================================================
# OUTPUT DASHBOARD
# ======================================================
st.markdown("---")
o1, o2, o3 = st.columns(3)

with o1:
    st.subheader("Implied Volatility")
    st.metric("IV (%)", f"{iv*100:.2f}%" if iv == iv else "N/A")
    st.metric("Implied Earnings Move", f"±{implied_move_pct*100:.2f}%")

with o2:
    st.subheader("Option Economics")
    st.metric("Call Value", f"{call_val:.2f}")
    st.metric("Put Value", f"{put_val:.2f}")

with o3:
    st.subheader("Asymmetry Signal")
    convexity = call_val / put_val if put_val > 0 else np.nan
    st.metric("Call / Put Ratio", f"{convexity:.2f}")

    if convexity > 1.2:
        st.success("Upside-Skewed Setup")
    elif convexity < 0.8:
        st.error("Downside-Skewed Setup")
    else:
        st.warning("Neutral Risk Profile")

# ======================================================
# EXPECTATION GAP
# ======================================================
st.markdown("---")
g1, g2 = st.columns(2)

with g1:
    upside_gap = upside/100 - implied_move_pct
    st.metric("Upside vs Market (Gap %)", f"{upside_gap*100:.2f}")

with g2:
    downside_gap = implied_move_pct - downside/100
    st.metric("Downside vs Market (Gap %)", f"{downside_gap*100:.2f}")

# ======================================================
# EARNINGS CALL TALK TRACK
# ======================================================
st.markdown("---")
st.subheader("📣 Earnings Call Talking Point")

st.info(
    f"""
    Options are pricing a **±{implied_move_pct*100:.1f}%** move into earnings.
    Our fundamental work implies **{upside:.1f}% upside** and **{downside:.1f}% downside**.
    This creates a **{'positive' if convexity > 1.2 else 'negative' if convexity < 0.8 else 'balanced'}**
    risk-reward skew going into the print.
    """
)

st.caption("Use for earnings-event expectation mapping — not long-term valuation.")
