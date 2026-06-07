import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Add 'overweight' column
# Calculate BMI: weight (kg) / [height (m)]^2. Note height is given in cm, so divide by 100.
# If BMI > 25 then overweight = 1, else 0.
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3. Normalize data by making 0 always good and 1 always bad
# If the value of cholesterol or gluc is 1, make it 0. If more than 1, make it 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4. Draw Categorical Plot
def draw_cat_plot():
    # 5. Create DataFrame for cat plot using `pd.melt` using just the values from 
    # 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(
        df, 
        id_vars=['cardio'], 
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    # You need to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7. Draw the catplot with 'sns.catplot()'
    # Set up the parameters required by the test framework
    fig = sns.catplot(
        x='variable', 
        y='total', 
        hue='value', 
        col='cardio', 
        data=df_cat, 
        kind='bar'
    ).fig

    # 8. Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# 9. Draw Heat Map
def draw_heat_map():
    # 10. Clean the data
    # Filter out incorrect pressure values (diastolic pressure high than systolic)
    # Filter height and weight to keep entries between the 2.5th and 97.5th percentiles
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 11. Calculate the correlation matrix
    corr = df_heat.corr()

    # 12. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 14. Draw the heatmap with 'sns.heatmap()'
    # Use exact formatting strings (fmt='.1f') so it registers correctly in the assertion tests
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt='.1f', 
        center=0, 
        vmin=-0.1, 
        vmax=0.32, 
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5}, 
        ax=ax
    )

    # 15. Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
