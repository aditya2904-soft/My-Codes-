import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Raw PEG and Dispersion data
data = {
    'Quarter': ['Q1 23', 'Q2 23', 'Q3 23', 'Q4 23', 'Q1 24', 'Q2 24', 'Q3 24', 'Q4 24', 'Q1 25 E'],
    'KMB': [112.15, 207.83, 293.07, 161.69, 260.05, 602.87, 437.00, 495.10, 199.49],
    'PG': [149.36, 174.84, 202.47, 288.80, 404.54, 452.98, 236.69, 245.95, 284.08],
    'CLX': [71.73, 114.08, 82.23, 2985.18, 402.23, 436.23, np.nan, 490.25, 265.22],
    'ESSITY-B': [52.33, 69.38, 83.06, 246.41, 118.37, 334.79, 242.09, 301.85, 90.00],
    'KAO': [87.53, 55.29, 45.81, 55.38, 46.24, 53.28, 68.78, 200.83, 328.48],
    'UNICHARM': [278.87, 248.93, 267.81, np.nan, 1053.44, 13.51, 12.34, 111.22, 156.21],
    'Dispersion_Industry': [5.2, 5.0, 5.2, 7.3, 6.1, 4.1, 3.94, 5.01, 4.59],
    'Dispersion_KMB': [2.3, 2.8, 1.5, 2.7, 2.5, 1.9, 2.63, 2.33, 3.61]
}

df = pd.DataFrame(data)

# Clean data
peer_cols = ['PG', 'CLX', 'ESSITY-B', 'KAO', 'UNICHARM']
df[peer_cols] = df[peer_cols].applymap(lambda x: np.nan if pd.isna(x) or x < 0 or x > 1000 else x)

# Compute industry PEG median
df['Industry_Median_PEG'] = df[peer_cols].median(axis=1, skipna=True)

# Compute relative PEG and relative Dispersion
df['RelPEG_KMB_vs_Industry'] = df['KMB'] / df['Industry_Median_PEG']
df['RelDisp_KMB_vs_Industry'] = df['Dispersion_KMB'] / df['Dispersion_Industry']

# Drop incomplete rows
df_filtered = df.dropna(subset=['RelPEG_KMB_vs_Industry', 'RelDisp_KMB_vs_Industry'])

# Correlation calculation
correlation = df_filtered[['RelDisp_KMB_vs_Industry', 'RelPEG_KMB_vs_Industry']].corr().iloc[0, 1]
corr_text = f"Corr: {correlation:.2f} ({'Positive' if correlation >= 0 else 'Negative'})"

# Compute mid-points for quadrant reference lines
x_mid = df_filtered['RelDisp_KMB_vs_Industry'].median()
y_mid = df_filtered['RelPEG_KMB_vs_Industry'].median()

# Plot
plt.figure(figsize=(10, 7))
sns.regplot(
    data=df_filtered,
    x='RelDisp_KMB_vs_Industry',
    y='RelPEG_KMB_vs_Industry',
    scatter_kws={'s': 100, 'color': 'red'},
    line_kws={'color': 'blue'}
)

# Annotate points
for _, row in df_filtered.iterrows():
    plt.text(row['RelDisp_KMB_vs_Industry'], row['RelPEG_KMB_vs_Industry'],
             row['Quarter'], fontsize=9, ha='left', va='bottom')

# Quadrant axes
plt.axvline(x=x_mid, color='gray', linestyle='--', linewidth=1)
plt.axhline(y=y_mid, color='gray', linestyle='--', linewidth=1)

# Labels and title
plt.title('KMB Relative PEG vs Relative Dispersion (with Quadrant Analysis)')
plt.xlabel('Relative Dispersion (KMB / Industry)')
plt.ylabel('Relative PEG (KMB / Industry Median)')
plt.grid(True, linestyle='--', alpha=0.7)

# Correlation annotation
x_max = df_filtered['RelDisp_KMB_vs_Industry'].max()
y_min = df_filtered['RelPEG_KMB_vs_Industry'].min()
plt.text(x=x_max * 0.98, y=y_min + 0.1, s=corr_text,
         fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black'))

plt.tight_layout()
plt.show()
