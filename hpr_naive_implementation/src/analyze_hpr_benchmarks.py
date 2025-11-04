import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Choose experiment type ---
root_dir = "../data/yannis_hpr/benchmarks_density"  # or ../data/yannis_hpr/benchmarks_distance
output_plot = os.path.join(root_dir, "summary_min_error.png")

# --- Collect all benchmark CSVs ---
csv_files = glob.glob(f"{root_dir}/**/benchmark_*.csv", recursive=True)
records = []

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    if "TOTAL_rate" not in df.columns:
        continue

    # --- Extract metadata from folder name ---
    parts = Path(csv_path).parts
    folder = Path(csv_path).parent.name
    # examples: bunny_1000_spherical_flip   OR   bunny_1x_exp_inversion
    tokens = folder.split("_")

    mesh = tokens[0]
    kernel = tokens[-1]
    # infer density or distance factor
    # Handle both density and distance-based naming
    variable_name = None
    variable_value = None

    # try:
    #     # Try parsing as a float (distance benchmark)
    #     variable_value = float(tokens[1])
    #     variable_name = "distance"
    # except ValueError:
    #     # Fallback to density benchmark
    try:
        variable_value = int(tokens[1])
        variable_name = "density"
    except ValueError:
        variable_value = float(tokens[1])
        variable_name = "density"

    print(f"[Parsed] mesh={mesh}, kernel={kernel}, {variable_name}={variable_value}")


    # --- Find minimum total error for this run ---
    min_row = df.loc[df["TOTAL_rate"].idxmin()]
    records.append({
        "mesh": mesh,
        "kernel": kernel,
        variable_name: variable_value,
        "min_error": min_row["TOTAL_rate"]
    })

# --- Combine all results ---
summary = pd.DataFrame(records)
print(summary.head())

# --- Save summary CSV ---
summary_csv = os.path.join(root_dir, "summary_min_errors.csv")
summary.to_csv(summary_csv, index=False)
print(f"[Saved] {summary_csv}")

# --- Plot results ---
plt.figure(figsize=(8, 6))

if "density" in summary.columns:
    xvar = "density"
    plt.xscale("log")
    xlabel = "Point Cloud Density (#points)"
else:
    xvar = "distance"
    xlabel = "Camera Distance (x mesh size)"

for (mesh, kernel), group in summary.groupby(["mesh", "kernel"]):
    plt.plot(group[xvar], group["min_error"], marker="o", linewidth=2, label=f"{mesh}-{kernel}")

# plt.xlabel(xlabel)
# plt.ylabel("Minimum Total Error (%)")
# plt.title("Minimum HPR Total Error vs " + ("Density" if xvar == "density" else "Camera Distance"))
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig(output_plot, dpi=300)
# plt.show()

g = sns.FacetGrid(summary, row="kernel", hue="mesh", height=3.5, aspect=2, sharey=True)
g.map(sns.lineplot, "density", "min_error", marker="o")
for ax in g.axes.flat:
    ax.set_xscale("log")
g.add_legend(title="Mesh")
g.set_axis_labels("density(× mesh size)", "Minimum Total Error (%)")
g.set_titles("Kernel: {row_name}")
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.show()


print(f"[Saved plot] {output_plot}")
