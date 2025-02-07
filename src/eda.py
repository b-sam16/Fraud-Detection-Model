import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

def univariate_analysis(df: pd.DataFrame, column: str):
    """
    Performs univariate analysis using histograms and boxplots.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")

    plt.show()

def bivariate_analysis(df: pd.DataFrame, x_column: str, y_column: str):
    """
    Performs bivariate analysis using scatterplots and correlation heatmaps.
    """
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df[x_column], y=df[y_column], alpha=0.5)
    plt.title(f"{x_column} vs {y_column}")
    plt.show()

    correlation = df[[x_column, y_column]].corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.title(f"Correlation between {x_column} and {y_column}")
    plt.show()

