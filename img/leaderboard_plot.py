import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Extract results from README
results = {}
with open('../README.md', 'r') as readme_file:
    readme = readme_file.read().split('\n')

# Find the last updated line
timestamp_line = next(line for line in readme if line.startswith('Last updated on'))
timestamp = timestamp_line.split('Last updated on ')[1]

# Extract performance table
table_start = readme.index('|----|----:|----:|----:|----:|')
table_end = readme.index('## Model')
performance_table = readme[table_start + 1: table_end - 1]

for record in performance_table:
    model_name = record.split('|')[1].strip()
    hallucination_rate = float(record.split('|')[2].replace(' %', ''))
    results[model_name] = hallucination_rate

# Construct dataframe
df = pd.DataFrame.from_dict(results, orient='index', columns=["hallucination_rate"])
df_sorted = df.sort_values(by='hallucination_rate', ascending=True).reset_index()
df_sorted.columns = ['LLM', 'hallucination_rate']
score_threshold = df_sorted.iloc[25]['hallucination_rate']
df_sorted = df_sorted[df_sorted['hallucination_rate'] <= score_threshold]

# Plot top 25 LLMs
fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(x='hallucination_rate', y='LLM', data=df_sorted,
                 palette='Spectral_r', orient='h')
fig.suptitle('Hallucination Rates for Top 25 LLMs', fontsize=15, y=1.03)

# Remove axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# Add value labels on the bars
for i, v in enumerate(df_sorted['hallucination_rate']):
    ax.text(v + 0.1, i, f'{v:.1f}%', va='center')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set x-axis limit
ax.set_xlim(0, 3.5)

# Add logo
logo_path = 'vectara_logo.png'
image = plt.imread(logo_path)
imagebox = OffsetImage(image, zoom=0.04)
annotation_box = AnnotationBbox(imagebox, (0.9, 1), frameon=False, xycoords='axes fraction', box_alignment=(1, 1))
# annotation_box = AnnotationBbox(imagebox, (2.5, len(df_sorted) + 0.5), frameon=False, xycoords='data', box_alignment=(0.5, 0))
ax.add_artist(annotation_box)

plt.figtext(0.5, 0.02, f"Last updated on {timestamp}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('hallucination_rates_with_logo.png', dpi=300, bbox_inches='tight')
plt.show()