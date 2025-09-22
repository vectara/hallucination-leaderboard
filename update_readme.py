import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import colorsys
import pandas as pd
from io import StringIO

# === Load leaderboard data ===
with open("output/stats_all_LLMs.json", "r") as f:
    raw_data = json.load(f)

# === Create DataFrame ===
df = pd.DataFrame(raw_data)

# Strip and convert percent-based fields (none have % in JSON but we normalize anyway)
# Round hallucination rates to one decimal
df["Hallucination Rate"] = df["hallucination_rate"].round(1)
df["Factual Consistency Rate"] = 100 - df["hallucination_rate"]
df["Answer Rate"] = df["answer_rate"].round(1)
df["Average Summary Length (Words)"] = df["avg_word_count"].round(1)
df["Model"] = df["model_name"]

# Sort by hallucination rate
df_sorted = df.sort_values("Hallucination Rate", ascending=True).reset_index(drop=True)
df_top10 = df_sorted.head(10)
df_top25 = df_sorted.head(25)

# === Generate Markdown Table ===
table_md = "|Model|Hallucination Rate|Factual Consistency Rate|Answer Rate|Average Summary Length (Words)|\n"
table_md += "|----|----:|----:|----:|----:|\n"
for _, row in df_top10.iterrows():
    table_md += f"|{row['Model']}|{row['Hallucination Rate']} %|{row['Factual Consistency Rate']} %|{row['Answer Rate']} %|{row['Average Summary Length (Words)']}|\n"

# === Generate Plot ===
def slightly_desaturate(color, factor=0.8):
    r, g, b, a = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r_d, g_d, b_d = colorsys.hls_to_rgb(h, l, s * factor)
    return (r_d, g_d, b_d, a)

num_colors = len(df_top25)
indices = np.linspace(0, 1, num_colors + 2)[2:]
turbo_colors = cm.turbo(indices)
colors = np.array([slightly_desaturate(c, factor=0.6) for c in turbo_colors])

fig, ax = plt.subplots(figsize=(14, 10))
bars = ax.barh(df_top25["Model"], df_top25["Hallucination Rate"], color=colors)
ax.set_xlabel("Hallucination Rate (%)")
fig.suptitle("Grounded Hallucination Rates for Top 25 LLMs", fontsize=24, x=0.45, y=0.95)
ax.invert_yaxis()

# Add bar labels
for i, value in enumerate(df_top25["Hallucination Rate"]):
    ax.text(value + 0.05, i, f"{value:.1f}%", va='center')

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(len(df_top25) - 0.5, -0.5)

# === Add logo ===
logo_path = "vectara_logo_official.png"
try:
    logo = Image.open(logo_path)
    scale = 0.5
    new_size = (int(logo.width * scale), int(logo.height * scale))
    logo_resized = logo.resize(new_size, Image.LANCZOS)
    logo_resized = np.array(logo_resized)
    fig.figimage(logo_resized, xo=3000, yo=2200, zorder=10, alpha=1.0)
except Exception as e:
    print(f"Logo not added: {e}")

# === Save plot ===
date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
plot_filename = f"top25_hallucination_rates_{date_str}.png"
plot_path = f"./img/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# === Format timestamp ===
date_for_readme = datetime.datetime.utcnow().strftime("%B %d, %Y")  # e.g., September 22, 2025

# === Compose leaderboard section for README ===
readme_section = f"""Last updated on {date_for_readme}

![Plot: hallucination rates of various LLMs]({plot_path})

{table_md}
"""

# === Insert section between tags ===
with open("README.md", "r") as f:
    readme = f.read()

start_tag = "<!-- LEADERBOARD_START -->"
end_tag = "<!-- LEADERBOARD_END -->"

if start_tag in readme and end_tag in readme:
    before = readme.split(start_tag)[0]
    after = readme.split(end_tag)[1]
    new_readme = before + start_tag + "\n" + readme_section + "\n" + end_tag + after
else:
    new_readme = readme + "\n\n" + start_tag + "\n" + readme_section + "\n" + end_tag

# === Save updated README ===
with open("README.md", "w") as f:
    f.write(new_readme)