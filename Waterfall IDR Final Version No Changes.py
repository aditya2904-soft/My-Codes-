import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.legend import Legend

# Specify the file path
file_path = r"D:\Waterfall PT File.xlsx"
sheet_name = 'Football Field Analysis'

# Load the Excel file and the specific sheet
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found. Please check the path and try again.")
    exit()
except Exception as e:
    print(f"Error loading the Excel file: {e}")
    exit()

# Extract the relevant data
# Categories are in column E (index 4), rows 2-6 (iloc[1:6])
categories = df.iloc[1:5, 4].tolist()

# Mins in column F (index 5)
mins = df.iloc[1:5, 5].tolist()

# Maxs in column H (index 7)
maxs = df.iloc[1:5, 7].tolist()

# Stock price from column I (index 8), any row (using row 2, iloc[1])
stock_price = df.iloc[1, 8]

# Check for valid data
if not categories or not mins or not maxs or pd.isna(stock_price):
    print("Error: Missing or invalid data in the required columns/rows.")
    exit()

# Extract price targets from column B (index 1), starting from row 3 (iloc[2:])
price_targets_df = df.iloc[2:, 1].dropna()
price_targets = pd.to_numeric(price_targets_df, errors='coerce').dropna().values

# Calculate ideal range (25th to 75th percentile) if there are enough data points exluding 52 week high/low 
# Exclude "52 Weeks High/low" from price_targets for ideal range calculation
filtered_targets = price_targets_df[~df.iloc[2:, 0].astype(str).str.contains("52 Weeks High/low", case=False, na=False)]
filtered_targets = pd.to_numeric(filtered_targets, errors='coerce').dropna().values

if len(filtered_targets) >= 4:  # Ensure there are enough data points
    ideal_min = np.percentile(filtered_targets, 25)
    ideal_max = np.percentile(filtered_targets, 75)
else:
    ideal_min = None
    ideal_max = None
    print("Not enough data points to calculate ideal range.")

# Reverse the lists to plot from top to bottom (optional for better visualization)
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

# Add ideal range overlay if calculated with values highlighted on x-axis
if ideal_min is not None and ideal_max is not None and ideal_min < ideal_max:
    # Overlay the ideal range as a shaded region
    ax.axvspan(ideal_min, ideal_max, alpha=0.25, color='lightblue', label=f'Ideal Range ({ideal_min:.2f} - {ideal_max:.2f})')
    # Highlight ideal min and max values on x-axis
    ax.axvline(ideal_min, color='blue', linestyle=':', linewidth=2)
    ax.axvline(ideal_max, color='blue', linestyle=':', linewidth=2)
    # Add text label for ideal range, rotated 90 degrees at the center of the ideal range
    ax.text((ideal_min + ideal_max) / 2, len(categories) / 2, 'Ideal Range',
            ha='center', va='center', rotation=90, color='blue', fontsize=12)

# Set y-ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(categories, fontsize=12)

# Set labels and title
ax.set_xlabel('Price Target ($)', fontsize=12)
ax.set_title('Football Field Valuation Chart for SMAR', fontsize=14)

# Add grid for better readability
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Add legend highlight for ideal range Values with transparency 
# Add legend highlight for ideal range Values with transparency 

handles, labels = ax.get_legend_handles_labels()
if ideal_min is not None and ideal_max is not None and ideal_min < ideal_max:
    handles.append(Patch(facecolor='lightblue', edgecolor='blue', alpha=0.25, label=f'Ideal Range ({ideal_min:.2f} - {ideal_max:.2f})')) ,

# Place legend below the chart
ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Adjust layout to prevent clipping
plt.tight_layout()



# Show the plot (comment out if running in a non-interactive environment)
plt.show()