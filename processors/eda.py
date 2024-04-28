import json
from tqdm import tqdm

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


with open(r'config/config.json') as f:
    config = json.load(f)


def pie_distrbution(df, target_col, savefile=r'figures/pie_chart.png', figsize=(16,5), palette='colorblind', name='Train'):
    df = df.fillna('Nan')

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.flatten()

    # Pie chart
    pie_colors = sns.color_palette(palette, len(df[target_col].unique()))
    ax[0].pie(
        df[target_col].value_counts(),
        shadow=True,
        explode=[0.05] * len(df[target_col].unique()),
        autopct='%1.f%%',
        textprops={'size': 15, 'color': 'white'},
        colors=pie_colors
    )
    ax[0].set_aspect('equal')  # Fix the aspect ratio to make the pie chart circular

    # Bar plot
    bar_colors = sns.color_palette(palette)
    sns.countplot(
        data=df,
        y=target_col,
        ax=ax[1],
        hue=target_col,
        legend=False,
        # palette=bar_colors
    )
    ax[1].set_xlabel('Count', fontsize=14)
    ax[1].set_ylabel('')
    ax[1].tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(width=0)  # Remove tick lines for y-axis

    fig.suptitle(f'{target_col} in {name} Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Show the plot
    plt.savefig(savefile)


def hot_encode(df: pd.DataFrame):
    one_hot_encoded_col_list = []
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False)


    for col in categorical_cols:
        if df[col].nunique()/len(df)<config["HOT_ENCODE_CATEGORY_LIMIT"]:
            one_hot_encoded_col_list.append(col)
    
    one_hot_encoded_data = encoder.fit_transform(df[one_hot_encoded_col_list])
    one_hot_df = pd.DataFrame(one_hot_encoded_data, columns=encoder.get_feature_names_out(one_hot_encoded_col_list))
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(one_hot_encoded_col_list, axis=1)
    
    return df_encoded


def plot_histograms(df, columns, savefile=r'figures/barplot.png', n_cols=3):
    df = df[columns]
    n_rows = (len(columns) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()

    for i, var_name in enumerate(df.columns.tolist()):
        if var_name != 'is_generated':
            ax = axes[i]
            sns.histplot(df[var_name], kde=True, ax=ax, label='Train')
            ax.set_title(f'{var_name} Distribution (Train vs Test)')
            ax.legend()

    # plt.tight_layout()
    plt.savefig(savefile)


def plot_distribution(df, columns, hue, savefile=r'figures/barplot_distribution.png', title="train data", drop_cols=[]):
    sns.set_style('whitegrid')

    df = df[columns]
    cols = df.columns.drop([hue] + drop_cols)
    n_cols = 2
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 4*n_rows))

    for i, var_name in enumerate(tqdm(cols)):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.histplot(data=df, x=var_name, kde=True, ax=ax, hue=hue) # sns.distplot(df_train[var_name], kde=True, ax=ax, label='Train')
        ax.set_title(f'{var_name} Distribution')

    fig.suptitle(f'{title} Distribution Plot by {hue}', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(savefile)


def dataframe_corr(df: pd.DataFrame, columns: list, savefile=r'figures/correlation.png', title="train data"):
    df = df[columns]

    # Create a mask for the diagonal elements
    mask = np.zeros_like(df.astype(float).corr())
    mask[np.triu_indices_from(mask)] = True

    # Set the colormap and figure size
    colormap = plt.cm.RdBu_r
    plt.figure(figsize=(15, 15))

    # Set the title and font properties
    plt.title(f'{title} Correlation of Features', fontweight='bold', y=1.02, size=20)

    # Plot the heatmap with the masked diagonal elements
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 10, "weight": "bold"},
                mask=mask)
    
    plt.savefig(savefile)


def plot_scatter_matrix(df: pd.DataFrame, columns: list, target, savefile=r'figures/graphical_correlation.png', size=26):
    # sns.pairplot()
    
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(len(columns), len(columns), figsize=(size, size), sharex=False, sharey=False)

    for i, col in enumerate(tqdm(columns)):
        for j, col_ in enumerate(columns):
            axes[i,j].set_xlabel(f'{col}', fontsize=14)
            axes[i,j].set_ylabel(f'{col_}', fontsize=14)

            # Plot the scatterplot
            sns.scatterplot(data=df, x=col, y=col_, hue=target, ax=axes[i,j],
                            s=80, edgecolor='gray', alpha=0.2, palette='bright')

            axes[i,j].tick_params(axis='both', which='major', labelsize=12)

            if i == 0:
                axes[i,j].set_title(f'{col_}', fontsize=18)
            if j == 0:
                axes[i,j].set_ylabel(f'{col}', fontsize=18)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.legend(loc='upper right', ncol=5, fontsize=18)
    plt.savefig(savefile)


def hierarchical_clustering(df, columns, target, savefile=r'figures/hierarchy.png', title='Train data'):
    df = df[columns.drop(target)]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=120)
    correlations = df.corr()
    converted_corr = 1 - np.abs(correlations)
    Z = linkage(squareform(converted_corr), 'complete')
    
    dn = dendrogram(Z, labels=df.columns, ax=ax, above_threshold_color='#ff0000', orientation='right')
    hierarchy.set_link_color_palette(None)
    plt.grid(axis='x')
    plt.title(f'{title} Hierarchical clustering, Dendrogram', fontsize=18, fontweight='bold')
    plt.savefig(savefile)


