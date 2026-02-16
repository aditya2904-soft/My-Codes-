import streamlit as st
import pandas as pd

# Load structured metrics (you can replace with parsed transcript data)
data = pd.DataFrame({
    "Company": ["PANW", "FTNT", "CHKP"],
    "Revenue": ["$10.00B", "$6.75B", "$2.71B"],
    "Growth": ["18%", "13.3%", "6%"],
    "Operating Margin": ["30%", "32.75%", "41%"],
    "EPS": ["$5.25 est", "$2.50 est", "$9.90 est"],
    "ThreatLevel": ["Leadership", "High Threat", "Medium Threat"]
})

st.set_page_config(page_title="PANW Competitive Intelligence Dashboard", layout="wide")

# Confidential Banner
st.markdown(
    "<div style='background:#dc2626;color:white;text-align:center;"
    "padding:8px;font-weight:700;letter-spacing:1px;'>"
    "🔒 CONFIDENTIAL - PALO ALTO NETWORKS INTERNAL USE ONLY</div>",
    unsafe_allow_html=True
)

# Header
st.title("Competitive Intelligence Dashboard")
st.subheader("Strategic Analysis of Key Cybersecurity Market Competitors")
st.caption("Q1–Q2 FY25 Briefing")

# Strategic Takeaways
with st.container():
    st.markdown("### 🚀 Strategic Takeaways")
    st.write("- Revenue Growth: FTNT outpaces CHKP")
    st.write("- Margin Pressure: Both face compression")
    st.write("- SaaS/SASE adoption: creating displacement opportunities")
    st.write("- AI spend: vulnerability window before payoff")

# Competitor Cards
for idx, row in enumerate(data.itertuples(index=False)):
    st.markdown(f"## {row.Company}")
    st.markdown(f"<span style='color: #dc2626; font-weight: 600;'>Threat Level: {row.ThreatLevel}</span>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Revenue", row.Revenue)
    cols[1].metric("Growth", row.Growth)
    cols[2].metric("Op Margin", row.Operating_Margin if hasattr(row, 'Operating_Margin') else row._asdict().get('Operating Margin', 'N/A'))
    cols[3].metric("EPS", row.EPS)
    if idx < len(data) - 1:
        st.divider()

# Recommendations
st.markdown("## ⚡ Strategic Recommendations")
st.write("1. Accelerate platform messaging against FTNT cannibalization + CHKP slow growth")
st.write("2. Use pricing flexibility vs. margin pressure")
st.write("3. Double down on SASE/AI-native differentiation")
st.write("4. Launch displacement campaigns into their refresh cycles")
