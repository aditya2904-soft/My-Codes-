import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# File path provided by the user
file_path = r"D:\PTM.xlsx"

# Function to extract bins and frequencies dynamically from a sheet
def extract_bins_freq(sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    # Find all positions of 'Bin'
    mask = (df == 'Bin')
    bin_locations = mask.stack()[mask.stack()].index.tolist()
    bin_row, bin_col = None, None
    # Find the 'Bin' where next column is 'Frequency'
    for row, col in bin_locations:
        if col + 1 < df.shape[1] and df.iloc[row, col + 1] == 'Frequency':
            bin_row = row
            bin_col = col
            break
    if bin_row is None:
        raise ValueError(f"No valid 'Bin' with adjacent 'Frequency' found in sheet {sheet_name}")
    # Extract data starting from next row until empty
    start_row = bin_row + 1
    bins_list = []
    freq_list = []
    for i in range(start_row, df.shape[0]):
        bin_val = pd.to_numeric(df.iloc[i, bin_col], errors='coerce')
        freq_val = pd.to_numeric(df.iloc[i, bin_col + 1], errors='coerce')
        if pd.notna(bin_val) and pd.notna(freq_val) and freq_val > 0:
            bins_list.append(bin_val)
            freq_list.append(freq_val)
        elif pd.isna(bin_val) and pd.isna(freq_val):
            break  # Stop if both are NaN
    return np.array(bins_list), np.array(freq_list)

# Extract for ideal range (Skewness ex outlier)
bins_ideal, freq_ideal = extract_bins_freq("Skewness ex outlier")

# Extract for full (Skewness)
bins_full, freq_full = extract_bins_freq("Skewness")

# Create side-by-side plots to match the image layout
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

import matplotlib.ticker as mticker

# Left plot: Skewness Excluding Outlier - Ideal Price Range (smoothed)
x_ideal = bins_ideal
y_ideal = freq_ideal
if len(x_ideal) > 2:
    xnew_ideal = np.linspace(x_ideal.min(), x_ideal.max(), 300)
    spl_ideal = make_interp_spline(x_ideal, y_ideal, k=3)
    ynew_ideal = spl_ideal(xnew_ideal)
    axs[0].plot(xnew_ideal, ynew_ideal, color='blue')
    axs[0].scatter(x_ideal, y_ideal, color='blue')
else:
    axs[0].plot(x_ideal, y_ideal, marker='o', linestyle='-', color='blue')
axs[0].set_title('Skewness Excluding Outlier - Ideal Price Range')
axs[0].set_xlabel('Price Target')
axs[0].set_ylabel('Brokers')
axs[0].set_xlim(210, 235)
axs[0].set_ylim(0, 5)
axs[0].grid(True)
axs[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Right plot: Skewness - Price Target (total brokers, smoothed)
x_full = bins_full
y_full = freq_full
if len(x_full) > 2:
    xnew_full = np.linspace(x_full.min(), x_full.max(), 300)
    spl_full = make_interp_spline(x_full, y_full, k=3)
    ynew_full = spl_full(xnew_full)
    axs[1].plot(xnew_full, ynew_full, color='blue')
    axs[1].scatter(x_full, y_full, color='blue')
else:
    axs[1].plot(x_full, y_full, marker='o', linestyle='-', color='blue')
axs[1].set_title('Skewness - Price Target')
axs[1].set_xlabel('Price Target')
axs[1].set_ylabel('Brokers')
axs[1].set_xlim(125, 245)
axs[1].set_ylim(0, 20)
axs[1].grid(True)
axs[1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Adjust layout and show/save the plot
plt.tight_layout()
# plt.savefig('PANW_Skewness_Charts.png')  # Uncomment to save instead of showing
plt.show()