import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# extract results from README
results = {}
readme = open('../README.md','r').read().split('\n')
table_start = readme.index('|----|----:|----:|----:|----:|')
table_end = readme.index('## Model')
performance_table = readme[table_start+1: table_end-1]
for record in performance_table:
    model_name = record.split('|')[1].split()[-1]
    hallucination_rate = float(record.split('|')[2].replace(' %',''))
    results[model_name] = hallucination_rate

# construct dataframe
df = pd.DataFrame.from_dict(results, orient='index', columns=["hallucination_rate"])
df_sorted = df.sort_values(by='hallucination_rate', ascending=True).reset_index()
df_sorted.columns = ['LLM','hallucination_rate']
score_threshold = df_sorted.iloc[25]['hallucination_rate']
df_sorted = df_sorted.loc[df_sorted['hallucination_rate'] <= score_threshold]

# plot top 25 LLMs
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='hallucination_rate', y='LLM', data=df_sorted,
                 palette='Spectral_r', orient='h')
plt.title('Hallucination Rate for Top 25 LLMs',pad=25)
ax.set_xlabel('')
ax.set_ylabel('')

# Add value labels on the bars
for i, v in enumerate(df_sorted['hallucination_rate']):
    ax.text(v + 0.1, i, f'{v:.1f}%', va='center')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set x-axis limit
ax.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('hallucination_rates_of_various_LLMs.png', dpi=300, bbox_inches='tight')
plt.show()

