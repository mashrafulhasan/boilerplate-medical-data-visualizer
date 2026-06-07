import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =============================================================
# 1. Import data
# =============================================================
df = pd.read_csv('medical_examination.csv')

# =============================================================
# 2. Add 'overweight' column
#    BMI = weight(kg) / (height(m))^2
#    BMI > 25  →  1 (overweight),  else  →  0
# =============================================================
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# =============================================================
# 3. Normalize cholesterol & gluc
#    Original value == 1  →  0  (good)
#    Original value  > 1  →  1  (bad)
# =============================================================
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc']        = (df['gluc']        > 1).astype(int)


# =============================================================
# 4. Categorical Plot
#    Tests expect:
#      - xlabel  == "variable"
#      - ylabel  == "total"
#      - x-ticks == ['active','alco','cholesterol','gluc','overweight','smoke']
#      - 13 Rectangle patches in axes[0]
# =============================================================
def draw_cat_plot():

    # Melt to long format; id_vars must be 'cardio'
    df_cat = pd.melt(
        df,
        id_vars='cardio',
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # Count occurrences per (cardio, variable, value); column MUST be named 'total'
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # Draw — col='cardio' splits into two panels; test reads axes[0]
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )
    fig = g.fig

    fig.savefig('catplot.png')
    return fig


# =============================================================
# 5. Heat Map
#    Tests expect:
#      - x-tick labels: ['id','age','sex','height','weight','ap_hi','ap_lo',
#                        'cholesterol','gluc','smoke','alco','active','cardio','overweight']
#      - 91 specific annotation strings (lower triangle only)
# =============================================================
def draw_heat_map():

    # --- Clean the data ---
    df_heat = df[
        (df['ap_lo']  <= df['ap_hi'])  &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # --- Correlation matrix (lower triangle) ---
    corr = df_heat.corr(numeric_only=True)

    # Mask upper triangle (including diagonal)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # --- Draw heatmap ---
    fig, ax = plt.subplots(figsize=(12, 9))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
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
