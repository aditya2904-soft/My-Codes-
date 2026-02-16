import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Step 1: Sample data input ---
data = {
    'Quarter': ['Q2 24', 'Q3 24', 'Q4 24', 'Q1 25'],
    'KMB': [706.32, 554.15, 337.38, 227.16],
    'Industry_Avg': [471.56, 428.54, 432.02, 435.45]  # Replace with your equal-weighted industry average
}
df = pd.DataFrame(data)
df.set_index('Quarter', inplace=True)

# --- Step 2: Normalize to Q2 24 (Indexed to 100) ---
df_indexed = df / df.iloc[0] * 100

# --- Plot 1: Indexed Time Series ---
plt.figure(figsize=(10, 5))
plt.plot(df_indexed, marker='o')
plt.title("Indexed Performance (Q2 24 = 100): KMB vs Industry Avg")
plt.ylabel("Index (Base = 100)")
plt.grid(True)
plt.legend(df_indexed.columns)
plt.tight_layout()
plt.show()

# --- Plot 2: Rolling Correlation (Not meaningful with only 4 data points, but for completeness) ---
if len(df) >= 4:
    rolling_corr = df['KMB'].rolling(4).corr(df['Industry_Avg'])
    plt.figure(figsize=(8, 4))
    plt.plot(rolling_corr, marker='o')
    plt.title("Rolling 4-Quarter Correlation: KMB vs Industry")
    plt.ylabel("Correlation Coefficient")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Plot 3: Static Correlation Scatter Plot + Regression ---
x = df['Industry_Avg'].values.reshape(-1, 1)
y = df['KMB'].values
model = LinearRegression().fit(x, y)
r_squared = model.score(x, y)

plt.figure(figsize=(6, 5))
sns.regplot(x='Industry_Avg', y='KMB', data=df)
plt.title(f"KMB vs Industry Avg\nR² = {r_squared:.2f}")
plt.grid(True)
plt.tight_layout()
plt.show()
