import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_heatmap(df, title, save_path=None, figsize=(10, 8), ):
    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap='viridis', annot=True, fmt=".1f", cbar=True,
                xticklabels=True, yticklabels=True, annot_kws={"fontsize": 6})
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_histogram(data, title, bins=50, save_path=None):
    if isinstance(data, (pd.Series, np.ndarray)):
        values = data
    else:
        values = np.array(data)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_density(data, title=None, xlabel="Distance", ylabel="Density", save_path=None):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data, fill=True, color="blue", linewidth=1.5)
    plt.title(title or "")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 200)  # Adjust range if needed
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_boxplot_by_class(df, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Class", y="Distance")
    plt.xticks(rotation=90)
    plt.title("Intra-Target Distance by Drug Class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()
