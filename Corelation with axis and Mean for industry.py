import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Step 1: Raw Data
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

# Step 2: Data Cleaning
peer_cols = ['PG', 'CLX', 'ESSITY-B', 'KAO', 'UNICHARM']
df[peer_cols] = df[peer_cols].applymap(lambda x: np.nan if pd.isna(x) or x < 0 or x > 1000 else x)

# Step 3: Compute Metrics Using MEAN
df['Industry_Mean_PEG'] = df[peer_cols].mean(axis=1, skipna=True)
df['RelPEG_KMB_vs_Industry'] = df['KMB'] / df['Industry_Mean_PEG']
df['RelDisp_KMB_vs_Industry'] = df['Dispersion_KMB'] / df['Dispersion_Industry']

# Step 4: Filter Valid Quarters
df_filtered = df.dropna(subset=['RelPEG_KMB_vs_Industry', 'RelDisp_KMB_vs_Industry'])

# Step 5: Correlation Analysis
correlation = df_filtered[['RelDisp_KMB_vs_Industry', 'RelPEG_KMB_vs_Industry']].corr().iloc[0, 1]
corr_text = f"Corr: {correlation:.2f} ({'Positive' if correlation >= 0 else 'Negative'})"

# Step 6: Quadrant Axes (Mean-Based)
x_mean = df_filtered['RelDisp_KMB_vs_Industry'].mean()
y_mean = df_filtered['RelPEG_KMB_vs_Industry'].mean()

# Step 7: Plotting
plt.figure(figsize=(12, 8))

# KMB Points
plt.scatter(df_filtered['RelDisp_KMB_vs_Industry'],
            df_filtered['RelPEG_KMB_vs_Industry'],
            color='red', s=100, label='KMB (per Quarter)')

# Linear Regression Line
sns.regplot(data=df_filtered,
            x='RelDisp_KMB_vs_Industry',
            y='RelPEG_KMB_vs_Industry',
            scatter=False,
            color='blue',
            label='Linear Fit')

# Quarter Labels
for _, row in df_filtered.iterrows():
    plt.text(row['RelDisp_KMB_vs_Industry'] + 0.005,
             row['RelPEG_KMB_vs_Industry'],
             row['Quarter'], fontsize=9, ha='left')

# Quadrant Lines
plt.axvline(x=x_mean, color='gray', linestyle='--', linewidth=1.5)
plt.axhline(y=y_mean, color='gray', linestyle='--', linewidth=1.5)

# Axis Labels
plt.xlabel('Relative Dispersion = KMB / Industry (Mean-based)\n', fontsize=10)
plt.ylabel('Relative PEG = KMB / Industry Mean\n', fontsize=10)

# Correlation Annotation
x_max = df_filtered['RelDisp_KMB_vs_Industry'].max()
y_max = df_filtered['RelPEG_KMB_vs_Industry'].max()
plt.text(x=x_max * 0.95, y=y_max * 0.95, s=corr_text,
         fontsize=12, ha='right', va='top',
         bbox=dict(facecolor='white', edgecolor='black'))

# Title & Legend
plt.title('Quadrant Plot: Relative PEG vs Relative Dispersion for KMB (Mean-Based)', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10, frameon=True)

# Grid & Layout
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
