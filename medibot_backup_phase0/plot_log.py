#!/usr/bin/env python3
"""Plot localisation error from the mission logs.

    python plot_log.py                -> compares every mission_*.csv in controllers/medibot_controller/logs
Run one mission with tag correction ON and one with it OFF (press T while IDLE) for the comparison slide.
"""
import csv
import glob
import os

import matplotlib.pyplot as plt

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controllers", "medibot_controller", "logs")
files = sorted(glob.glob(os.path.join(LOGS, "mission_*.csv")))
if not files:
    raise SystemExit("No logs yet - run at least one mission in Webots first.")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for fn in files:
    rows = list(csv.DictReader(open(fn)))
    if not rows:
        continue
    on = rows[0]["tags_on"] == "1"
    label = ("AprilTag correction ON" if on else "AprilTag correction OFF") + f" ({os.path.basename(fn)[8:14]})"
    t0 = float(rows[0]["t"])
    t = [float(r["t"]) - t0 for r in rows]
    a1.plot(t, [float(r["err_cm"]) for r in rows], label=label, color="tab:green" if on else "tab:red")
    a2.plot([float(r["true_x"]) for r in rows], [float(r["true_y"]) for r in rows],
            label=label + " - true path", color="tab:green" if on else "tab:red")
a1.set_xlabel("time (s)")
a1.set_ylabel("position error of estimate (cm)")
a1.set_title("Localisation error")
a1.grid(alpha=.3)
a1.legend()
a2.set_xlabel("x (m)")
a2.set_ylabel("y (m)")
a2.set_title("Path actually driven")
a2.set_aspect("equal")
a2.grid(alpha=.3)
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("localisation_comparison.png", dpi=150)
print("saved localisation_comparison.png")
plt.show()
