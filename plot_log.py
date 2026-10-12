#!/usr/bin/env python3
"""Plot recorded MediBot localisation error and driven/planned paths.

Run ``py plot_log.py`` after one or more missions. The PNG is written beside this
script as ``localisation_comparison.png`` at 200 dpi.
"""
import csv
import glob
import os

import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "controllers", "medibot_controller", "logs")
OUTPUT = os.path.join(ROOT, "localisation_comparison.png")
files = sorted(glob.glob(os.path.join(LOGS, "mission_*.csv")))
if not files:
    raise SystemExit("No mission logs yet; run at least one mission in Webots first.")

fig, (error_ax, path_ax) = plt.subplots(1, 2, figsize=(13, 5.5))
plotted = 0
for index, filename in enumerate(files):
    with open(filename, newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    if not rows:
        continue

    tags_on = rows[0].get("tags_on", "1") == "1"
    tag_label = "tags ON" if tags_on else "tags OFF"
    color = "tab:green" if tags_on else "tab:red"
    linestyle = "-" if tags_on else "--"
    label = f"{tag_label} ({index + 1})"
    t0 = float(rows[0]["t"])
    times = [float(row["t"]) - t0 for row in rows]
    errors = [float(row["err_cm"]) for row in rows]
    error_ax.plot(times, errors, color=color, linestyle=linestyle,
                  alpha=0.82, label=label)

    driven_x = [float(row["true_x"]) for row in rows]
    driven_y = [float(row["true_y"]) for row in rows]
    path_ax.plot(driven_x, driven_y, color=color, linestyle="-", linewidth=1.8,
                 alpha=0.85, label=f"{label} - driven")

    # Target waypoints are sampled with each mission pose row. Compress repeated
    # target values to recover the planned node sequence from the log.
    planned = []
    for row in rows:
        px, py = row.get("planned_x", ""), row.get("planned_y", "")
        if not px or not py:
            continue  # Older logs predate planned waypoint columns.
        point = (float(px), float(py))
        if not planned or point != planned[-1]:
            planned.append(point)
    if planned:
        path_ax.plot([p[0] for p in planned], [p[1] for p in planned],
                     color=color, linestyle="--", linewidth=1.5,
                     alpha=0.75, label=f"{label} - planned")
    plotted += 1

if plotted == 0:
    raise SystemExit("Mission log files were found, but none contained samples.")

error_ax.set_xlabel("time (s)")
error_ax.set_ylabel("position error (cm)")
error_ax.set_title("Pose error over time (tags ON / OFF)")
error_ax.grid(alpha=0.3)
error_ax.legend(fontsize=8)

path_ax.set_xlabel("x (m)")
path_ax.set_ylabel("y (m)")
path_ax.set_title("Driven path and planned waypoints")
path_ax.set_aspect("equal", adjustable="datalim")
path_ax.grid(alpha=0.3)
path_ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=200)
plt.close(fig)
print(f"Saved {OUTPUT} ({plotted} mission log(s), 200 dpi)")
