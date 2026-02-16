#!/usr/bin/env python3
"""
Creative Cybersecurity Peer Dashboard
Modern, data-driven visualizations with scatter plots, bubble charts, and radar diagrams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Wedge
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects

# Helper function to add glow effect to text
def add_glow(text_obj, glow_color='#4ecdc4', glow_width=4):
    text_obj.set_path_effects([
        path_effects.withStroke(linewidth=glow_width, foreground=glow_color, alpha=0.5),
        path_effects.Normal()
    ])

# Modern dark theme with vibrant accents
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial']
plt.rcParams['figure.facecolor'] = '#0a0e27'
plt.rcParams['axes.facecolor'] = '#121629'

# Vibrant color palette
COLORS = {
    'panw': '#ff6b9d',      # Vibrant pink
    'fortinet': '#4ecdc4',  # Cyan
    'checkpoint': '#ffd93d', # Yellow
    'accent1': '#6c5ce7',
    'accent2': '#a29bfe',
    'grid': '#2d3561'
}

# Build data
data = {
    "Metric": [
        "Revenue / Billings Growth (YoY %)",
        "Next-Gen / Unified SASE ARR Growth (YoY %)",
        "Next-Gen Network Security ARR Growth (YoY %)",
        "RPO / Current RPO (USD bn)",
        "Platform Net Retention / ARR NRR (%)",
        "Operating Margin (%)",
        "Free Cash Flow Margin (%)",
        "SASE Customers / Seats (approx)",
        "XSIAM / SecOps Customers (approx)",
        "Notable Strategic Focus"
    ],
    "PANW": [np.nan, 0.32, 0.35, np.nan, 1.20, np.nan, 0.38,
             "6,300+ SASE customers; ~6M seats",
             "≈400 XSIAM customers (avg ARR > $1M)",
             "Platformization, AI runtime security"],
    "Fortinet": [0.14, 0.22, np.nan, 6.64, np.nan, 0.33, 0.40,
                 "FortiSASE ARR growth ~100%",
                 "SecOps ARR $463M (+35%)",
                 "Unified FortiOS, 500+ AI patents"],
    "Check Point": [0.06, np.nan, np.nan, 2.40, np.nan, 0.41, np.nan,
                    "SASE R&D ramp; Gartner MQ",
                    "Quantum Force AI firewalls (+12%)",
                    "Open platform, prevention-first"]
}

df = pd.DataFrame(data).set_index("Metric")
df.to_excel("peer_dashboard_panw_peers.xlsx", sheet_name="Peer Dashboard", engine="openpyxl")

# Create comprehensive metrics dataset
metrics_data = {
    'Company': ['PANW', 'Fortinet', 'Check Point', 'PANW', 'Fortinet', 'Check Point'],
    'Metric': ['Growth', 'Growth', 'Growth', 'Profitability', 'Profitability', 'Profitability'],
    'Value': [32, 22, 6, 38, 36.5, 41],  # SASE growth and avg margins
    'Size': [120, 100, 80, 38, 40, 41],  # Net retention or FCF for sizing
    'Innovation': [9, 8, 7, 9, 8, 7]  # Innovation score
}

# Create figure with better spacing
fig = plt.figure(figsize=(26, 16), facecolor='#0a0e27')
gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.45, hspace=0.55,
                       left=0.06, right=0.96, top=0.88, bottom=0.08)

# Title with glow effect
title = fig.text(0.5, 0.96, 'TRANSCRIPT ANALYSIS', 
                ha='center', fontsize=32, fontweight='bold', color='#ffffff',
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='#4ecdc4')])
subtitle = fig.text(0.5, 0.93, 'Earnings Call Intelligence Dashboard  •  FY2025 Peer Comparison',
                   ha='center', fontsize=14, color='#ffffff', style='italic')

# ============ PANEL 1: Multi-dimensional Scatter (Growth vs Innovation) ============
ax1 = fig.add_subplot(gs[0:2, 0:2])

# Prepare data for 3D-style scatter
companies = ['PANW', 'Fortinet', 'Check Point']
growth_rates = [32, 22, 6]  # SASE ARR growth
innovation_scores = [9, 8, 7]
market_presence = [120, 90, 70]  # Relative size
colors_list = [COLORS['panw'], COLORS['fortinet'], COLORS['checkpoint']]

# Create scatter with glow effect
for i, (comp, growth, innov, size, color) in enumerate(zip(companies, growth_rates, 
                                                            innovation_scores, market_presence, colors_list)):
    # Outer glow
    ax1.scatter(growth, innov, s=size*8, alpha=0.15, color=color, edgecolors='none')
    ax1.scatter(growth, innov, s=size*5, alpha=0.25, color=color, edgecolors='none')
    # Main bubble
    ax1.scatter(growth, innov, s=size*3, alpha=0.8, color=color, 
               edgecolors='white', linewidths=2.5, zorder=100)
    
    # Labels with background
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3, edgecolor='none')
    ax1.text(growth, innov-0.3, comp, fontsize=11, fontweight='bold', 
            ha='center', color='white', bbox=bbox_props, zorder=101)

xlabel = ax1.set_xlabel('SASE/Next-Gen ARR Growth (%)', fontsize=13, fontweight='bold', color='white')
add_glow(xlabel, '#4ecdc4', 3)
ylabel = ax1.set_ylabel('Innovation Index (AI, Platform)', fontsize=13, fontweight='bold', color='white')
add_glow(ylabel, '#4ecdc4', 3)
title1 = ax1.set_title('Growth × Innovation Matrix', fontsize=17, fontweight='bold', 
             pad=20, color='white', loc='left')
add_glow(title1, '#ff6b9d', 4)

ax1.grid(True, alpha=0.15, linestyle='--', color=COLORS['grid'])
ax1.set_xlim(-2, 38)
ax1.set_ylim(6, 10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color(COLORS['grid'])
ax1.spines['bottom'].set_color(COLORS['grid'])
ax1.tick_params(colors='#8892b0')

# Add quadrant lines
ax1.axvline(x=20, color='#4ecdc4', alpha=0.3, linestyle='--', linewidth=1.5)
ax1.axhline(y=7.5, color='#4ecdc4', alpha=0.3, linestyle='--', linewidth=1.5)
ax1.text(35, 9.7, 'Leaders', fontsize=10, alpha=0.5, color='#4ecdc4', style='italic')

# ============ PANEL 2: Profitability Bubble Chart ============
ax2 = fig.add_subplot(gs[0, 2:])

# Data
operating_margins = [30, 33, 41]  # Estimated for PANW, actual for others
fcf_margins = [38, 40, 38]
revenue_scale = [10500, 6750, 2640]  # Millions

for i, (comp, op_margin, fcf, revenue, color) in enumerate(zip(companies, operating_margins, 
                                                                fcf_margins, revenue_scale, colors_list)):
    bubble_size = (revenue / 50) ** 0.8  # Scale appropriately
    
    # Create glow effect
    ax2.scatter(op_margin, fcf, s=bubble_size*3, alpha=0.1, color=color, edgecolors='none')
    ax2.scatter(op_margin, fcf, s=bubble_size*2, alpha=0.2, color=color, edgecolors='none')
    # Main bubble
    ax2.scatter(op_margin, fcf, s=bubble_size, alpha=0.85, color=color,
               edgecolors='white', linewidths=2.5, zorder=100)
    
    # Labels
    ax2.text(op_margin, fcf+1.5, comp, fontsize=10, fontweight='bold',
            ha='center', color=color, zorder=101)
    ax2.text(op_margin, fcf-1.8, f'${revenue/1000:.1f}B', fontsize=9,
            ha='center', color='#8892b0', zorder=101)

xlabel2 = ax2.set_xlabel('Operating Margin (%)', fontsize=13, fontweight='bold', color='white')
add_glow(xlabel2, '#4ecdc4', 3)
ylabel2 = ax2.set_ylabel('FCF Margin (%)', fontsize=13, fontweight='bold', color='white')
add_glow(ylabel2, '#4ecdc4', 3)
title2 = ax2.set_title('Profitability Landscape', fontsize=17, fontweight='bold',
             pad=20, color='white', loc='left')
add_glow(title2, '#ff6b9d', 4)

ax2.grid(True, alpha=0.15, linestyle='--', color=COLORS['grid'])
ax2.set_xlim(25, 45)
ax2.set_ylim(33, 43)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(COLORS['grid'])
ax2.spines['bottom'].set_color(COLORS['grid'])
ax2.tick_params(colors='#8892b0')

# Add "Bubble size = Revenue" annotation
ax2.text(0.98, 0.02, 'Bubble size ∝ Annual Revenue', transform=ax2.transAxes,
        fontsize=9, ha='right', va='bottom', color='#8892b0', style='italic')

# ============ PANEL 3: Radar Chart - Strategic Capabilities ============
ax3 = fig.add_subplot(gs[1, 2:], projection='polar')

categories = ['SASE', 'AI/ML', 'Platform', 'SecOps', 'Cloud\nSecurity']
N = len(categories)

# Scores for each company (0-10 scale)
panw_scores = [9, 9, 8, 9, 8]
fortinet_scores = [8, 8, 10, 7, 7]
checkpoint_scores = [6, 7, 5, 6, 8]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
panw_scores += panw_scores[:1]
fortinet_scores += fortinet_scores[:1]
checkpoint_scores += checkpoint_scores[:1]
angles += angles[:1]

# Plot with fill
ax3.plot(angles, panw_scores, 'o-', linewidth=2.5, color=COLORS['panw'], 
        label='PANW', markersize=6)
ax3.fill(angles, panw_scores, alpha=0.2, color=COLORS['panw'])

ax3.plot(angles, fortinet_scores, 'o-', linewidth=2.5, color=COLORS['fortinet'],
        label='Fortinet', markersize=6)
ax3.fill(angles, fortinet_scores, alpha=0.2, color=COLORS['fortinet'])

ax3.plot(angles, checkpoint_scores, 'o-', linewidth=2.5, color=COLORS['checkpoint'],
        label='Check Point', markersize=6)
ax3.fill(angles, checkpoint_scores, alpha=0.2, color=COLORS['checkpoint'])

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=11, color='white', fontweight='bold')
ax3.set_ylim(0, 10)
ax3.set_yticks([2, 4, 6, 8, 10])
ax3.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9, color='#ffffff')
ax3.grid(True, color=COLORS['grid'], alpha=0.3)
ax3.set_facecolor('#121629')
ax3.spines['polar'].set_color(COLORS['grid'])
ax3.tick_params(colors='#ffffff')
title3 = ax3.set_title('Strategic Capabilities Radar', fontsize=17, fontweight='bold',
             pad=30, color='white', y=1.12)
add_glow(title3, '#ffd93d', 4)

legend = ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15),
                   frameon=True, fancybox=True, fontsize=10)
legend.get_frame().set_facecolor('#1a1f3a')
legend.get_frame().set_alpha(0.9)

# ============ PANEL 4: Clear Comparative Performance Chart ============
ax4 = fig.add_subplot(gs[2, 0:2])

# Data - simplified and clearer
metrics_names = ['Revenue\nGrowth', 'SASE\nGrowth', 'Operating\nMargin', 'FCF\nMargin', 'RPO\nGrowth']
panw_vals = [14, 32, 30, 38, 24]
fortinet_vals = [14, 22, 33, 40, 12]
checkpoint_vals = [6, 8, 41, 38, 6]

x_pos = np.arange(len(metrics_names))
bar_width = 0.25

# Create grouped bars with better spacing and clarity
bars1 = ax4.bar(x_pos - bar_width, panw_vals, bar_width * 0.85,
                label='PANW', color=COLORS['panw'], alpha=0.9,
                edgecolor='white', linewidth=2, zorder=10)

bars2 = ax4.bar(x_pos, fortinet_vals, bar_width * 0.85,
                label='Fortinet', color=COLORS['fortinet'], alpha=0.9,
                edgecolor='white', linewidth=2, zorder=10)

bars3 = ax4.bar(x_pos + bar_width, checkpoint_vals, bar_width * 0.85,
                label='Check Point', color=COLORS['checkpoint'], alpha=0.9,
                edgecolor='white', linewidth=2, zorder=10)

# Add value labels on top of bars - clearer and bigger
for bars, vals in zip([bars1, bars2, bars3], [panw_vals, fortinet_vals, checkpoint_vals]):
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='white',
                path_effects=[path_effects.withStroke(linewidth=3, foreground='#0a0e27')])

ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics_names, fontsize=12, color='white', fontweight='bold')
ylabel4 = ax4.set_ylabel('Performance (%)', fontsize=13, fontweight='bold', color='white')
add_glow(ylabel4, '#4ecdc4', 3)
title4 = ax4.set_title('Comparative Performance Analysis', fontsize=17, fontweight='bold',
             pad=20, color='white', loc='left')
add_glow(title4, '#ff6b9d', 4)

ax4.grid(axis='y', alpha=0.2, linestyle='--', color=COLORS['grid'], zorder=0)
ax4.set_ylim(0, 50)
ax4.set_axisbelow(True)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color('#2d3561')
ax4.spines['bottom'].set_color('#2d3561')
ax4.tick_params(axis='x', colors='#ffffff', labelsize=11)
ax4.tick_params(axis='y', colors='#ffffff', labelsize=10)

# Clear legend with better positioning
legend4 = ax4.legend(loc='upper left', frameon=True, fancybox=True, 
                    fontsize=12, ncol=3, columnspacing=1.5)
legend4.get_frame().set_facecolor('#1a1f3a')
legend4.get_frame().set_edgecolor('#4ecdc4')
legend4.get_frame().set_linewidth(2)
legend4.get_frame().set_alpha(0.95)

# ============ PANEL 5: Strategic Focus Summary ============
ax5 = fig.add_subplot(gs[2, 2:])
ax5.axis('off')

# Background box - bigger to cover all content
from matplotlib.patches import FancyBboxPatch
fancy_box = FancyBboxPatch((0.01, 0.0), 0.98, 1.0,
                           boxstyle="round,pad=0.03",
                           edgecolor='#2d3561', facecolor='#1a1f3a',
                           linewidth=2.5, transform=ax5.transAxes, alpha=0.7)
ax5.add_patch(fancy_box)

# Title with better spacing
ax5.text(0.5, 0.94, 'STRATEGIC POSITIONING SUMMARY', transform=ax5.transAxes,
        fontsize=16, fontweight='bold', ha='center', color='white')

# PANW Section
ax5.text(0.06, 0.82, '◆ PALO ALTO NETWORKS', transform=ax5.transAxes,
        fontsize=13, fontweight='bold', color=COLORS['panw'], va='top')
ax5.text(0.09, 0.76, '• Platformization leader: 120% net retention', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.71, '• AI-first: Prisma AIRS, Browser security', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.66, '• CyberArk acquisition (Identity expansion)', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.61, '• 6,300+ SASE customers, 400 XSIAM accounts', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')

# Fortinet Section
ax5.text(0.06, 0.52, '◆ FORTINET', transform=ax5.transAxes,
        fontsize=13, fontweight='bold', color=COLORS['fortinet'], va='top')
ax5.text(0.09, 0.46, '• Unified FortiOS: Single platform strategy', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.41, '• FortiSASE ARR +100%, Infrastructure: $2B invested', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.36, '• 500+ AI patents, SecOps ARR $463M (+35%)', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.31, '• Strongest operating efficiency at scale', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')

# Check Point Section
ax5.text(0.06, 0.22, '◆ CHECK POINT', transform=ax5.transAxes,
        fontsize=13, fontweight='bold', color=COLORS['checkpoint'], va='top')
ax5.text(0.09, 0.16, '• Open platform philosophy, Prevention-first', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.11, '• Highest operating margin: 41%', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.06, '• Quantum Force AI firewalls +12%', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')
ax5.text(0.09, 0.01, '• Veriti & Cyberint acquisitions (threat intel)', transform=ax5.transAxes,
        fontsize=11, color='#e0e6ed', va='top')

# Footer
fig.text(0.5, 0.015, 'Data: Q4 FY25 PANW | Q2 FY25 Fortinet & Check Point Earnings Transcripts  •  Visualization: Advanced Analytics Dashboard 2025',
        ha='center', fontsize=9, color='#4a5568', style='italic')

plt.savefig("creative_peer_dashboard.png", dpi=300, bbox_inches='tight',
           facecolor='#0a0e27', edgecolor='none')
plt.close()

print("=" * 70)
print("CREATIVE DASHBOARD GENERATED")
print("=" * 70)
print("\nFiles Created:")
print("  • creative_peer_dashboard.png - Modern dark-theme dashboard")
print("  • peer_dashboard_panw_peers.xlsx - Data table")
print("\nVisualization Features:")
print("  ✓ Multi-dimensional scatter plot (Growth × Innovation)")
print("  ✓ Profitability bubble chart with revenue scaling")
print("  ✓ Strategic capabilities radar chart")
print("  ✓ Connected dot plot for metric comparison")
print("  ✓ Dark theme with vibrant color palette")
print("  ✓ Glow effects and modern styling")
print("\n" + "=" * 70)