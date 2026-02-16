import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

# Original numeric data
raw_data = {
    'Quarter': ['Q1 23', 'Q2 23', 'Q3 23', 'Q4 23', 'Q1 24', 'Q2 24', 'Q3 24', 'Q4 24', 'Q1 25'],
    'KMB': [-322.43, -295.04, -357.50, -347.88, -785.46, 735.87, 539.47, 320.37, 248.21],
    'PG': [589.00, 525.34, 539.69, 492.06, 505.97, 481.10, 448.94, 420.42, 456.05],
    'CLX': [-240.17, -248.06, -157.83, -183.26, -221.41, -433.31, 237.21, 255.55, 135.72],
    'ESSITY-B': [104.27, 73.86, 19.03, 58.84, 258.41, 46.26, 244.27, -580.11, 769.07],
    'KAO': [-130.72, -127.88, -139.84, -172.75, -161.87, -154.87, -196.17, -269.12, -507.34],
    'UNICHARM': [3944.44, 383.59, 307.99, 519.28, 488.44, -857.30, -670.62, -471.85, -2440.86],
    'Dispersion - Industry Equal Wt.': [5.1, 5.1, 4.9, 7.6, 5.9, 3.9, 4.12, 5.23, 3.33],
    'Dispersion - KMB': [2.1, 4.1, 2.0, 3.2, 2.7, 5.3, 2.70, 1.71, 3.60]
}

# Create DataFrame from raw data
df_numeric = pd.DataFrame(raw_data)

# Create display version with negative values replaced by '-'
df_display = df_numeric.copy()
for col in ['KMB', 'PG', 'CLX', 'ESSITY-B', 'KAO', 'UNICHARM']:
    df_display[col] = df_display[col].apply(lambda x: '-' if x < 0 else x)

# Now prepare df_cleaned for plotting: replace negatives with NaN
df_cleaned = df_numeric.copy()
for col in ['KMB', 'PG', 'CLX', 'ESSITY-B', 'KAO', 'UNICHARM']:
    df_cleaned[col] = df_cleaned[col].apply(lambda x: np.nan if x < 0 else x)

# Melt for plotting
df_long = pd.melt(df_cleaned, id_vars=['Quarter'], 
                  value_vars=['KMB', 'PG', 'CLX', 'ESSITY-B', 'KAO', 'UNICHARM'],
                  var_name='Company', value_name='Peg Level')

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(x='Quarter', y='Peg Level', data=df_long, palette='Blues', ax=ax1)
ax1.scatter(df_numeric['Quarter'], df_cleaned['KMB'], color='red', label='KMB', s=100, zorder=5)
box_legend = mpatches.Patch(color='lightblue', label='Price to sales Group level (Boxplot)')

# Second Y-axis for dispersion
ax2 = ax1.twinx()
ax2.plot(df_numeric['Quarter'], df_numeric['Dispersion - Industry Equal Wt.'], label='Industry Dispersion (%)', color='green', marker='o', linewidth=2)
ax2.plot(df_numeric['Quarter'], df_numeric['Dispersion - KMB'], label='KMB Dispersion (%)', color='orange', marker='o', linewidth=2)

# Formatting
ax1.set_title('Peg Level Distribution and KMB Trend with Dispersion on t - 30')
ax1.set_xlabel('Quarter')
ax1.set_ylabel('Peg Level')
ax2.set_ylabel('Dispersion (%)')
ax2.set_ylim(0, 10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax2.grid(False)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines1 = [box_legend] + lines1
labels1 = [box_legend.get_label()] + labels1
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


