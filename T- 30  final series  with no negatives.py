import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

# Clean numeric data from image (negative values kept for cleaning later)
raw_data = {
    'Quarter': ['Q1 23', 'Q2 23', 'Q3 23', 'Q4 23', 'Q1 24', 'Q2 24', 'Q3 24', 'Q4 24', 'Q1 25 E'],
    'KMB': [112.15, 207.83, 293.07, 161.69, 260.05, 602.87, 437.00, 495.10, 199.49],
    'PG': [149.36, 174.84, 202.47, 288.80, 404.54, 452.98, 236.69, 245.95, 284.08],
    'CLX': [71.73, 114.08, 82.23, 2985.18, 402.23, 436.23, -343.16, 490.25, 265.22],
    'ESSITY-B': [52.33, 69.38, 83.06, 246.41, 118.37, 334.79, 242.09, 301.85, 90.00],
    'KAO': [87.53, 55.29, 45.81, 55.38, 46.24, 53.28, 68.78, 200.83, 328.48],
    'UNICHARM': [278.87, 248.93, 267.81, -705.19, 1053.44, 13.51, 12.34, 111.22, 156.21],
    'Dispersion - Industry Equal Wt.': [5.2, 5.0, 5.2, 7.3, 6.1, 4.1, 3.94, 5.01, 4.59],
    'Dispersion - KMB': [2.3, 2.8, 1.5, 2.7, 2.5, 1.9, 2.63, 2.33, 3.61]
}

# Convert to DataFrame
df_numeric = pd.DataFrame(raw_data)

# Clean negative PEG values for boxplot (set to NaN)
df_cleaned = df_numeric.copy()
for col in ['KMB', 'PG', 'CLX', 'ESSITY-B', 'KAO', 'UNICHARM']:
    df_cleaned[col] = df_cleaned[col].apply(lambda x: np.nan if x < 0 else x)

# Prepare long-form DataFrame for Seaborn
df_long = pd.melt(df_cleaned, id_vars='Quarter',
                  value_vars=['KMB', 'PG', 'CLX', 'ESSITY-B', 'KAO', 'UNICHARM'],
                  var_name='Company', value_name='Peg Level')

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(x='Quarter', y='Peg Level', data=df_long, palette='Blues', ax=ax1)

# Plot KMB's individual trend as red dots
ax1.scatter(df_numeric['Quarter'], df_cleaned['KMB'], color='red', label='KMB', s=100, zorder=5)

# Manual legend for boxplot
box_legend = mpatches.Patch(color='lightblue', label='Price-to-Sales Group Level (Boxplot)')

# Secondary Y-axis for dispersion trends
ax2 = ax1.twinx()
ax2.plot(df_numeric['Quarter'], df_numeric['Dispersion - Industry Equal Wt.'], label='Industry Dispersion (%)', color='green', marker='o', linewidth=2)
ax2.plot(df_numeric['Quarter'], df_numeric['Dispersion - KMB'], label='KMB Dispersion (%)', color='orange', marker='o', linewidth=2)

# Titles and labels
ax1.set_title('PEG Level Distribution and KMB Trend with Dispersion Overlay at T- 30')
ax1.set_xlabel('Quarter')
ax1.set_ylabel('PEG Level')
ax2.set_ylabel('Dispersion (%)')
ax2.set_ylim(0, 10)

# Grid and legend setup
ax1.grid(True, linestyle='--', alpha=0.7)
ax2.grid(False)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines1 = [box_legend] + lines1
labels1 = [box_legend.get_label()] + labels1
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Final layout tweaks
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
