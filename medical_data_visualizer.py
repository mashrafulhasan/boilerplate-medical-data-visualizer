import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------
# 1. Import data
# -------------------------------------------------------
df = pd.read_csv('medical_examination.csv')

# -------------------------------------------------------
# 2. Add 'overweight' column
#    BMI = weight(kg) / (height(m))^2
#    BMI > 25 → 1 (overweight), else → 0
# -------------------------------------------------------
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# -------------------------------------------------------
# 3. Normalize cholesterol and gluc
#    value == 1  → 0 (good)
#    value  > 1  → 1 (bad)
# -------------------------------------------------------
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc']        = (df['gluc']        > 1).astype(int)


# -------------------------------------------------------
# 4. Categorical Plot
# -------------------------------------------------------
def draw_cat_plot():
    # Melt into long format
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # Group by cardio + variable + value, count rows, name column 'total'
    df_cat = df_cat.groupby(
        ['cardio', 'variable', 'value']
    ).size().reset_index(name='total')

    # Draw catplot
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    fig.savefig('catplot.png')
    return fig


# -------------------------------------------------------
# 5. Heat Map
# -------------------------------------------------------
def draw_heat_map():
    # Clean data — filter out incorrect rows
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Correlation matrix
    corr = df_heat.corr(numeric_only=True)

    # Mask upper triangle (True = hidden)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 9))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='0.1f',       # matches expected labels like '0.0', '-0.1', '0.5'
        linewidths=0.5,
        square=True,
        center=0,
        vmin=-0.16,
        vmax=0.32,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    fig.savefig('heatmap.png')
    return fig
