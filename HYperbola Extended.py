# Running the hyperbola band classification on the PT dataset for all four scenarios (A/B/C/D).
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Input PTs (14 values)
pt = np.array([45,55,43,60,50,55,55,50,42,55,52,57,55,55])
names = [f"Broker_{i+1}" for i in range(len(pt))]

# Multiples for scenarios from earlier interaction
mult_A = np.array([5,6,5,5,5,6,6,5,4,6,5,7,5,6])
mult_B = np.array([4,5,6,5,5,6,7,4,5,6,5,7,5,6])
mult_C = np.full_like(pt, 6)
q1, q2, q3 = np.percentile(pt, [25,50,75])
mult_D = np.zeros_like(pt)
mult_D[pt <= q1] = 4
mult_D[(pt > q1) & (pt <= q2)] = 5
mult_D[(pt > q2) & (pt <= q3)] = 6
mult_D[pt > q3] = 7
mult_D = mult_D.astype(int)

scenarios = {
    "A_same_as_before": mult_A,
    "B_example_new_mapping": mult_B,
    "C_single_multiple_6x": mult_C,
    "D_auto_quartile_mapping": mult_D
}

def hyperbola(x, a, b):
    return a + b / x

def fit_band(multiples, pts):
    # create df
    df = pd.DataFrame({"Multiple": multiples, "PT": pts})
    # group and get percentiles per multiple
    grouped = df.groupby("Multiple")["PT"].apply(list).reset_index(name="vals")
    grouped["P25"] = grouped["vals"].apply(lambda v: np.percentile(v,25))
    grouped["P50"] = grouped["vals"].apply(lambda v: np.percentile(v,50))
    grouped["P75"] = grouped["vals"].apply(lambda v: np.percentile(v,75))
    
    xdata = grouped["Multiple"].values.astype(float)
    p25 = grouped["P25"].values.astype(float)
    p50 = grouped["P50"].values.astype(float)
    p75 = grouped["P75"].values.astype(float)
    
    # Fit hyperbola curves where possible. If insufficient unique x, fallback to interpolation.
    fitted = {}
    try:
        params25, _ = curve_fit(hyperbola, xdata, p25, maxfev=5000)
        params50, _ = curve_fit(hyperbola, xdata, p50, maxfev=5000)
        params75, _ = curve_fit(hyperbola, xdata, p75, maxfev=5000)
        fitted["type"] = "hyperbola"
        fitted["params25"] = params25
        fitted["params50"] = params50
        fitted["params75"] = params75
    except Exception as e:
        # fallback: will use linear interpolation across xdata
        fitted["type"] = "interp"
        fitted["xdata"] = xdata
        fitted["p25"] = p25
        fitted["p50"] = p50
        fitted["p75"] = p75
    
    fitted["grouped"] = grouped
    return fitted

def eval_band_at(fitted, x_points):
    # return P25,P50,P75 evaluated at x_points (array-like)
    x = np.array(x_points, dtype=float)
    if fitted["type"] == "hyperbola":
        p25 = hyperbola(x, *fitted["params25"])
        p50 = hyperbola(x, *fitted["params50"])
        p75 = hyperbola(x, *fitted["params75"])
    else:
        # interpolation fallback
        p25 = np.interp(x, fitted["xdata"], fitted["p25"])
        p50 = np.interp(x, fitted["xdata"], fitted["p50"])
        p75 = np.interp(x, fitted["xdata"], fitted["p75"])
    return p25, p50, p75

# Run through scenarios and classify each broker
results_tables = {}
for name, mults in scenarios.items():
    fitted = fit_band(mults, pt)
    p25_at_brokers, p50_at_brokers, p75_at_brokers = eval_band_at(fitted, mults)
    
    df_out = pd.DataFrame({
        "Broker": names,
        "Multiple": mults,
        "PT": pt,
        "Band_P25_at_M": np.round(p25_at_brokers,3),
        "Band_P50_at_M": np.round(p50_at_brokers,3),
        "Band_P75_at_M": np.round(p75_at_brokers,3),
    })
    # classification and distance measures
    df_out["Inside_Band"] = ((df_out["PT"] >= df_out["Band_P25_at_M"]) & (df_out["PT"] <= df_out["Band_P75_at_M"]))
    # distance from median as % of median
    df_out["Pct_diff_vs_median"] = np.round((df_out["PT"] - df_out["Band_P50_at_M"]) / df_out["Band_P50_at_M"] * 100,2)
    # tag direction
    df_out["Status"] = df_out.apply(lambda r: "Inside" if r["Inside_Band"] else ("Above 75%" if r["PT"]>r["Band_P75_at_M"] else "Below 25%"), axis=1)
    
    results_tables[name] = {"fitted": fitted, "table": df_out}

# Display consolidated results for each scenario using a local display helper
def display_dataframe_to_user(title, df):
    # simple console display: print title and dataframe
    print(f"\n{title}")
    print(df.to_string(index=False))

for name, payload in results_tables.items():
    display_dataframe_to_user(f"Classification - {name}", payload["table"])
plt.show()

# Also save CSVs to /mnt/data and provide file paths
saved_files = []
for name, payload in results_tables.items():
    path = f"/mnt/data/classification_{name}.csv"
    payload["table"].to_csv(path, index=False)
    saved_files.append(path)

saved_files
