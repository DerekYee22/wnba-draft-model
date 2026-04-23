import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/derekyee/Documents/WNBA draft model claude/ncaaw_players_with_archetypes_ranked.csv")
df = df.dropna(subset=["pca1", "pca2", "archetype"])

BG = "#0d3d55"

ARCHETYPE_COLORS = {
    "Floor General":  "#4fc3f7",
    "3-and-D Wing":   "#81c784",
    "Interior Big":   "#ff8a65",
    "Post Scorer":    "#ce93d8",
    "Stretch Big":    "#ffd54f",
    "Combo Guard":    "#80deea",
}

# Wide crop — slide bottom banner proportions
fig, ax = plt.subplots(figsize=(16, 4), facecolor=BG)
ax.set_facecolor(BG)
ax.set_axis_off()

for archetype, color in ARCHETYPE_COLORS.items():
    sub = df[df["archetype"] == archetype]
    ax.scatter(sub["pca1"], sub["pca2"], c=color, s=10, alpha=0.45, linewidths=0)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = "/Users/derekyee/Documents/WNBA draft model claude/reports/pca_archetypes.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0)
print(f"Saved → {out}")
