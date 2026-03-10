"""
Generate all figures for the earthquake prediction journal paper.

Usage:
    cd earthquake-prediction
    python paper/generate_figures.py

Outputs PNG figures to paper/figures/
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "Earthquake_data_processed.csv")
MODEL_CMP = os.path.join(os.path.dirname(__file__), "..", "model_prep", "output", "model_comparison.csv")
FINAL_CMP = os.path.join(os.path.dirname(__file__), "..", "model_prep", "output", "final_comparison.csv")
FEAT_IMP = os.path.join(os.path.dirname(__file__), "..", "model_prep", "output", "feature_importance.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Magnitude Distribution
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_magnitude_distribution():
    df = pd.read_csv(DATA)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(df["Magnitude(ergs)"], bins=60, color="#2196F3", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Frequency")
    ax.set_title("(a) Distribution of Earthquake Magnitudes")
    ax.axvline(df["Magnitude(ergs)"].mean(), color="red", linestyle="--", linewidth=1, label=f'Mean = {df["Magnitude(ergs)"].mean():.2f}')
    ax.axvline(df["Magnitude(ergs)"].median(), color="orange", linestyle="--", linewidth=1, label=f'Median = {df["Magnitude(ergs)"].median():.2f}')
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_magnitude_distribution.png"))
    plt.close(fig)
    print("  ✓ fig1_magnitude_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Spatial Distribution of Epicenters
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_spatial_distribution():
    df = pd.read_csv(DATA)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(
        df["Longitude(deg)"], df["Latitude(deg)"],
        c=df["Magnitude(ergs)"], cmap="YlOrRd", s=2, alpha=0.5, edgecolors="none"
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Magnitude")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("(b) Spatial Distribution of Epicenters (Color = Magnitude)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_spatial_distribution.png"))
    plt.close(fig)
    print("  ✓ fig2_spatial_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Depth vs Magnitude
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_depth_vs_magnitude():
    df = pd.read_csv(DATA)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(df["Depth(km)"], df["Magnitude(ergs)"], s=2, alpha=0.3, color="#4CAF50", edgecolors="none")
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Magnitude")
    ax.set_title("(c) Hypocentral Depth vs. Magnitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_depth_vs_magnitude.png"))
    plt.close(fig)
    print("  ✓ fig3_depth_vs_magnitude.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Temporal Distribution (events per year)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_temporal_distribution():
    df = pd.read_csv(DATA)
    df["year"] = pd.to_datetime(df["Date(YYYY/MM/DD)"], format="%Y/%m/%d").dt.year
    yearly = df.groupby("year").agg(count=("Magnitude(ergs)", "size"), mean_mag=("Magnitude(ergs)", "mean"))
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.bar(yearly.index, yearly["count"], color="#2196F3", alpha=0.7, label="Event Count")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Events", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2 = ax1.twinx()
    ax2.plot(yearly.index, yearly["mean_mag"], color="red", linewidth=1.5, marker="o", markersize=3, label="Mean Magnitude")
    ax2.set_ylabel("Mean Magnitude", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax1.set_title("(d) Annual Earthquake Frequency and Mean Magnitude")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_temporal_distribution.png"))
    plt.close(fig)
    print("  ✓ fig4_temporal_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Model Comparison (Test R²)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_model_comparison():
    df = pd.read_csv(MODEL_CMP)
    df = df.sort_values("Test R²", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#F44336" if v < 0 else "#4CAF50" if v > 0.3 else "#FF9800" if v > 0.15 else "#2196F3" for v in df["Test R²"]]
    bars = ax.barh(df["Model"], df["Test R²"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Test R² Score")
    ax.set_title("(e) Model Comparison — Test R² (70/30 Split)")
    ax.axvline(0, color="black", linewidth=0.5)
    for bar, val in zip(bars, df["Test R²"]):
        ax.text(val + 0.01 if val >= 0 else val - 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_model_comparison.png"))
    plt.close(fig)
    print("  ✓ fig5_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Tuned vs Initial Performance
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_tuned_vs_initial():
    df = pd.read_csv(FINAL_CMP)
    metrics = ["R2", "MAE", "RMSE"]
    labels = ["R²", "MAE", "RMSE"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 3))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        x = np.arange(len(df))
        colors = ["#4CAF50" if "(tuned)" in name else "#2196F3" for name in df["Model"]]
        ax.barh(x, df[m], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(x)
        ax.set_yticklabels([n.replace(" (tuned)", "\n(tuned)").replace(" (initial)", "\n(initial)") for n in df["Model"]], fontsize=7)
        ax.set_xlabel(lab)
        ax.grid(axis="x", alpha=0.3)
        for j, val in enumerate(df[m]):
            ax.text(val + 0.002, j, f"{val:.4f}", va="center", fontsize=7)
    fig.suptitle("(f) Tuned vs. Initial Model Performance", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_tuned_vs_initial.png"))
    plt.close(fig)
    print("  ✓ fig6_tuned_vs_initial.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Feature Importance (Top 20)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_feature_importance():
    df = pd.read_csv(FEAT_IMP).head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.barh(df["feature"], df["importance"], color="#FF9800", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("(g) Top 20 Feature Importances — Extra Trees (Tuned)")
    for bar, val in zip(ax.patches, df["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_feature_importance.png"))
    plt.close(fig)
    print("  ✓ fig7_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8: Feature Category Contribution (pie)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_feature_categories():
    df = pd.read_csv(FEAT_IMP)
    temporal = ["year", "month", "day_of_year", "hour", "day_of_week",
                "month_sin", "month_cos", "hour_sin", "hour_cos",
                "dow_sin", "dow_cos", "time_since_prev", "log_time_since_prev"]
    spatial = ["lat_rad", "lon_rad", "dist_from_center", "lat_bin", "lon_bin",
               "spatial_cell", "Latitude(deg)", "Longitude(deg)"]
    seismo = ["log_depth", "stations_per_gap", "close_depth_interaction",
              "rms_log", "gap_close_ratio", "Depth(km)", "No_of_Stations",
              "Gap", "Close", "RMS"]
    rolling_lag = [f for f in df["feature"] if any(k in f for k in ["rolling", "lag", "diff", "spatial_event"])]
    encoded = ["Magnitude_type_enc"]

    cats = {"Temporal": temporal, "Spatial": spatial, "Seismological": seismo,
            "Rolling/Lag": rolling_lag, "Encoded": encoded}
    totals = {}
    for cat, feats in cats.items():
        totals[cat] = df[df["feature"].isin(feats)]["importance"].sum()

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    wedges, texts, autotexts = ax.pie(
        totals.values(), labels=totals.keys(), autopct="%1.1f%%",
        colors=colors, startangle=90, textprops={"fontsize": 9}
    )
    ax.set_title("(h) Feature Category Contribution to Total Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_feature_categories.png"))
    plt.close(fig)
    print("  ✓ fig8_feature_categories.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9: Correlation Heatmap of Raw Features
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_correlation_heatmap():
    df = pd.read_csv(DATA)
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation")
    ax.set_title("(i) Correlation Heatmap of Raw Numerical Features")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_correlation_heatmap.png"))
    plt.close(fig)
    print("  ✓ fig9_correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 10: Metric Comparison Radar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_metric_radar():
    df = pd.read_csv(MODEL_CMP)
    top5 = df.nlargest(5, "Test R²")
    categories = ["Test R²", "Test MAE", "Test RMSE", "Test MAPE (%)", "Explained Var"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    colors_list = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for idx, (_, row) in enumerate(top5.iterrows()):
        values = []
        for c in categories:
            v = row[c]
            if c in ["Test MAE", "Test RMSE", "Test MAPE (%)"]:
                v = 1 - min(v, 1)  # invert so larger = better
            values.append(v)
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=row["Model"], color=colors_list[idx], markersize=4)
        ax.fill(angles, values, alpha=0.1, color=colors_list[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["R²", "1−MAE", "1−RMSE", "1−MAPE%", "Expl. Var"], fontsize=8)
    ax.set_title("(j) Performance Radar — Top 5 Models", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig10_metric_radar.png"))
    plt.close(fig)
    print("  ✓ fig10_metric_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Run All
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating journal paper figures...")
    fig1_magnitude_distribution()
    fig2_spatial_distribution()
    fig3_depth_vs_magnitude()
    fig4_temporal_distribution()
    fig5_model_comparison()
    fig6_tuned_vs_initial()
    fig7_feature_importance()
    fig8_feature_categories()
    fig9_correlation_heatmap()
    fig10_metric_radar()
    print(f"\nAll figures saved to {OUT}/")
