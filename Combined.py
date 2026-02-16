import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# File path provided by the user
file_path = r"D:\PANW - SUPER FOLDER\PANW\panw\PANW Price Target Analysis.xlsx"

# Function to extract bins and frequencies dynamically from a sheet
def extract_bins_freq(sheet_name):
    try:
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
    except Exception as e:
        print(f"Error processing sheet {sheet_name}: {e}")
        return None, None

# --- Skewness Histograms ---
# Extract data for skewness plots
bins_ideal, freq_ideal = extract_bins_freq("Skewness ex outlier")
bins_full, freq_full = extract_bins_freq("Skewness")

# Create figure for side-by-side skewness plots
if bins_ideal is not None and bins_full is not None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

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

    # Adjust layout and save/show the plot
    plt.tight_layout()
    output_skewness_path = r"C:\Users\aditya.shivhare\Downloads\PANW_Skewness_Charts.png"
    plt.savefig(output_skewness_path, dpi=300)
    print(f"Skewness charts saved to: {output_skewness_path}")
    plt.show()
else:
    print("Skipping skewness plots due to data extraction errors.")

# --- Football Field Chart ---
# Load the Football Field Analysis sheet
sheet_name = 'Football Field Analysis'
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found. Please check the path and try again.")
    exit()
except Exception as e:
    print(f"Error loading the Excel file: {e}")
    exit()

# Extract the relevant data
categories = df.iloc[1:6, 4].tolist()  # Column E, rows 2-6
mins = df.iloc[1:6, 5].tolist()       # Column F, rows 2-6
maxs = df.iloc[1:6, 7].tolist()       # Column H, rows 2-6
stock_price = df.iloc[1, 8]           # Column I, row 2

# Check for valid data
if not categories or not mins or not maxs or pd.isna(stock_price):
    print("Error: Missing or invalid data in the required columns/rows for Football Field chart.")
    exit()

# Extract price targets from column B (index 1), starting from row 3 (iloc[2:])
price_targets_df = df.iloc[2:, 1].dropna()
price_targets = pd.to_numeric(price_targets_df, errors='coerce').dropna().values

# Calculate ideal range (25th to 75th percentile) excluding 52-week high/low
filtered_targets = price_targets_df[~df.iloc[2:, 0].astype(str).str.contains("52 Weeks High/low", case=False, na=False)]
filtered_targets = pd.to_numeric(filtered_targets, errors='coerce').dropna().values

if len(filtered_targets) >= 4:  # Ensure enough data points
    ideal_min = np.percentile(filtered_targets, 25)
    ideal_max = np.percentile(filtered_targets, 75)
else:
    ideal_min = None
    ideal_max = None
    print("Not enough data points to calculate ideal range.")

# Reverse lists for top-to-bottom plotting
categories = categories[::-1]
mins = mins[::-1]
maxs = maxs[::-1]

# Create the Football Field chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot horizontal lines for the min-max ranges
y_pos = range(len(categories))
ax.hlines(y_pos, mins, maxs, color='skyblue', linewidth=10, alpha=0.8)

# Add markers for min and max
ax.plot(mins, y_pos, 'o', color='blue', markersize=8, label='Min')
ax.plot(maxs, y_pos, 'o', color='navy', markersize=8, label='Max')

# Add a dashed vertical line for the current stock price
ax.axvline(x=stock_price, color='red', linestyle='--', linewidth=2, label=f'Current Stock Price ({stock_price:.2f})')

# Add ideal range overlay if calculated
if ideal_min is not None and ideal_max is not None and ideal_min < ideal_max:
    ax.axvspan(ideal_min, ideal_max, alpha=0.25, color='lightblue', label=f'Ideal Range ({ideal_min:.2f} - {ideal_max:.2f})')
    ax.axvline(ideal_min, color='blue', linestyle=':', linewidth=2)
    ax.axvline(ideal_max, color='blue', linestyle=':', linewidth=2)
    ax.text((ideal_min + ideal_max) / 2, len(categories) / 2, 'Ideal Range',
            ha='center', va='center', rotation=90, color='blue', fontsize=12)

# Set y-ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(categories, fontsize=12)

# Set labels and title
ax.set_xlabel('Price Target ($)', fontsize=12)
ax.set_title('Football Field Valuation Chart for PANW', fontsize=14)

# Add grid for better readability
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Add legend
handles, labels = ax.get_legend_handles_labels()
if ideal_min is not None and ideal_max is not None and ideal_min < ideal_max:
    handles.append(Patch(facecolor='lightblue', edgecolor='blue', alpha=0.25))
    labels.append(f'Ideal Range ({ideal_min:.2f} - {ideal_max:.2f})')
ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Adjust layout and save/show the plot
plt.tight_layout()
output_football_path = r"C:\Users\aditya.shivhare\Downloads\panw_football_field_chart.png"
plt.savefig(output_football_path, dpi=300)
print(f"Football field chart saved to: {output_football_path}")
plt.show()
