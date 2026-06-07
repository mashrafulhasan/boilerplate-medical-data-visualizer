g = sns.catplot(
    data=df_cat,
    x="variable",
    y="total",
    hue="value",
    col="cardio",
    kind="bar"
)
