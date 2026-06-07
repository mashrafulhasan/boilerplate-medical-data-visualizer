import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. Import data
# ---------------------------------------------------------------------------
df = pd.read_csv('medical_examination.csv')

# ---------------------------------------------------------------------------
# 2. Add 'overweight' column
#    BMI = weight(kg) / (height(m))^2
#    BMI > 25  →  overweight = 1,  else 0
# ---------------------------------------------------------------------------
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# ---------------------------------------------------------------------------
# 3. Normalize cholesterol and gluc
#    value == 1  →  0 (good)
#    value  > 1  →  1 (bad)
# ---------------------------------------------------------------------------
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc']        = (df['gluc']        > 1).astype(int)


# ---------------------------------------------------------------------------
# 4. Categorical Plot
# ---------------------------------------------------------------------------
def draw_cat_plot():
    # 4a. Melt the dataframe into long format
    df_cat = pd.melt(
        df,
        id_vars='cardio',
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 4b. Group and count, rename count column to 'total'
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # 4c. Draw the catplot
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


# ---------------------------------------------------------------------------
# 5. Heat Map
# ---------------------------------------------------------------------------
def draw_heat_map():
    # 5a. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi'])                                    # diastolic <= systolic
        & (df['height'] >= df['height'].quantile(0.025))                # height ≥ 2.5th percentile
        & (df['height'] <= df['height'].quantile(0.975))                # height ≤ 97.5th percentile
        & (df['weight'] >= df['weight'].quantile(0.025))                # weight ≥ 2.5th percentile
        & (df['weight'] <= df['weight'].quantile(0.975))                # weight ≤ 97.5th percentile
    ]

    # 5b. Correlation matrix (upper triangle only)
    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 5c. Draw heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        linewidths=0.5,
        center=0,
        vmin=-0.16,
        vmax=0.32,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    fig.savefig('heatmap.png')
    return fig
