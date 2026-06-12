"""Figure for the OCP-vs-sampling comparison (paper): planner success rate and planning time on the
narrow-passage transport query, with the OCP planning time as a reference.

Reads results/sampling_compare.json if present (written by sampling_compare.py --out), else uses the
embedded numbers from the reference run (timeout 20 s, 10 trials, seed 42). The OCP time is passed in.

Run (either env with matplotlib):
  python src/planner/plot_compare.py --ocp-time <seconds>
Output: results/sampling_vs_ocp.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# planner -> family, for colouring. The narrow-passage story is families, not individual planners.
FAMILY = {
    "RRT": "single-tree", "KPIECE1": "single-tree",
    "RRTConnect": "bidirectional", "BKPIECE1": "bidirectional", "LBKPIECE1": "bidirectional",
    "PRM": "roadmap", "PRMstar": "roadmap", "BITstar": "asymptotically-optimal",
}
FCOLOR = {"single-tree": "#d65f5f", "bidirectional": "#3a923a",
          "roadmap": "#e1812c", "asymptotically-optimal": "#3274a1"}

# Reference run (sampling_compare.py --sweep --timeout 20 --trials 10 --seed 42), query B_transport.
# success, t_med (s, over successes), paths_valid "good/n_success".
EMBEDDED_B = {
    "RRT":       (0.30, 1.646, "3/3"),
    "RRTConnect": (1.00, 0.553, "10/10"),
    "KPIECE1":   (0.50, 5.800, "5/5"),
    "BKPIECE1":  (1.00, 1.065, "10/10"),
    "LBKPIECE1": (1.00, 0.623, "10/10"),
    "PRM":       (0.70, 3.613, "6/7"),
    "PRMstar":   (0.80, 5.062, "7/8"),
    "BITstar":   (1.00, 3.186, "10/10"),
}
ORDER = ["RRT", "KPIECE1", "RRTConnect", "BKPIECE1", "LBKPIECE1", "PRM", "PRMstar", "BITstar"]


def load_B():
    here = os.path.dirname(__file__)
    jp = os.path.abspath(os.path.join(here, os.pardir, os.pardir, "results", "sampling_compare.json"))
    if os.path.exists(jp):
        d = json.load(open(jp))
        out = {}
        for r in d["records"]:
            if r["query"] == "B_transport":
                out[r["planner"]] = (r["success"], r["t_med"] or float("nan"),
                                     f"{r['paths_valid']}/{r['n_success']}")
        return out, d.get("timeout"), d.get("trials")
    return EMBEDDED_B, 20, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocp-time", type=float, required=True, help="OCP solve time [s]")
    ap.add_argument("--ocp-total", type=float, default=None, help="OCP total incl. build [s]")
    args = ap.parse_args()

    B, timeout, trials = load_B()
    names = [p for p in ORDER if p in B]
    colors = [FCOLOR[FAMILY[p]] for p in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: success rate over trials (the narrow-passage signature)
    succ = [B[p][0] * 100 for p in names]
    ax1.bar(range(len(names)), succ, color=colors)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("success rate over trials [%]")
    ax1.set_title(f"Narrow-passage transport: success rate\n(14-DoF whole-body, {trials} trials, "
                  f"{timeout}s timeout)")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=35, ha="right")
    ax1.axhline(100, color="0.6", lw=0.8, ls=":")
    for i, p in enumerate(names):
        ax1.text(i, succ[i] + 1.5, f"{succ[i]:.0f}%", ha="center", va="bottom", fontsize=8)

    # Panel 2: median time-to-first-solution (log), with the OCP reference
    tmed = [B[p][1] for p in names]
    ax2.bar(range(len(names)), tmed, color=colors)
    ax2.set_yscale("log")
    ax2.set_ylabel("median time-to-first-solution [s]  (over successes)")
    ax2.set_title("Planning time: sampling planners vs the OCP")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=35, ha="right")
    ax2.axhline(args.ocp_time, color="black", lw=1.8, ls="--",
                label=f"OCP solve = {args.ocp_time:.1f}s (feasible trajectory)")
    if args.ocp_total:
        ax2.axhline(args.ocp_total, color="0.4", lw=1.2, ls=":",
                    label=f"OCP total incl. build = {args.ocp_total:.1f}s")
    for i, p in enumerate(names):
        ax2.text(i, tmed[i] * 1.08, f"{tmed[i]:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FCOLOR.values()]
    fig.legend(handles, list(FCOLOR), loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sampling-based whole-body planning vs OCP on the box-transport narrow passage "
                 "(identical bounding-box geometry)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                                       "results", "sampling_vs_ocp.png"))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
