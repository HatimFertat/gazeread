import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
from pandas.plotting import parallel_coordinates

# Data
data = {
    'Researcher': ['A', 'B', 'C', 'D'],
    'Publications in ACM': [80, 20, 60, 50],
    'Publications in TVCG': [30, 0, 5, 8],
    'Publications in IEEE': [100, 10, 20, 100],
    'Paper Reviews': [500, 20, 200, 300],
    'Emails per Day': [300, 15, 150, 120]
}
df = pd.DataFrame(data)

# Stacked Bar Chart
def stacked_bar_chart(df):
    df.set_index('Researcher').plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('Researcher Activities - Stacked Bar Chart')
    plt.xlabel('Researcher')
    plt.ylabel('Counts')
    plt.legend(title='Attributes')
    plt.show()

# Bubble Chart
def bubble_chart(df):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df['Publications in ACM'], df['Publications in IEEE'], 
        s=df['Paper Reviews']*0.1, c=df['Emails per Day'], 
        cmap='viridis', alpha=0.6, edgecolors="w", linewidth=2
    )
    plt.xlabel('Publications in ACM')
    plt.ylabel('Publications in IEEE')
    plt.title('Researcher Activities - Bubble Chart')
    plt.colorbar(scatter, label='Emails per Day')
    for i, txt in enumerate(df['Researcher']):
        plt.annotate(txt, (df['Publications in ACM'][i], df['Publications in IEEE'][i]), fontsize=12, ha='right')
    plt.show()

# Matrix Plot
def matrix_plot(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.set_index('Researcher').T, annot=True, cmap='coolwarm')
    plt.title('Researcher Activities - Matrix Plot')
    plt.show()

# Radar (Spider) Chart
def radar_chart(df):
    categories = ['Publications in ACM', 'Publications in TVCG', 'Publications in IEEE', 'Paper Reviews', 'Emails per Day']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    
    for i, researcher in enumerate(df['Researcher']):
        values = df.loc[i].drop('Researcher').values.flatten().tolist()
        values += values[:1]
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, color='grey', size=16)
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=researcher)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Researcher Activities - Radar Chart')
    plt.show()
radar_chart(df)

# Parallel Coordinates Plot
def parallel_coordinates_plot(df):
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df, 'Researcher', colormap='viridis')
    plt.title('Researcher Activities - Parallel Coordinates Plot')
    plt.xlabel('Attributes')
    plt.ylabel('Values')
    plt.show()

# Heatmap with Collaboration Network
# Heatmap with Collaboration Network (Directed Edges)
def heatmap_with_network(df):
    plt.figure(figsize=(12, 6))
    
    # Heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(df.set_index('Researcher').T, annot=True, cmap='coolwarm')
    plt.title('Researcher Activities - Heatmap')
    
    # Network
    plt.subplot(1, 2, 2)
    G = nx.DiGraph()
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'A'), ('B', 'D'), ('C', 'A'), ('C', 'B'), ('D', 'A'), ('D', 'B'), ('D', 'C')]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', linewidths=1, font_size=15, arrowsize=20)
    plt.title('Researcher Collaboration Network (Directed)')
    
    plt.tight_layout()
    plt.show()

heatmap_with_network(df)
# Execute all plots
""" stacked_bar_chart(df)
bubble_chart(df)
matrix_plot(df)
radar_chart(df)
parallel_coordinates_plot(df)
heatmap_with_network(df) """


def line_chart(df):
    df_melted = pd.melt(df, id_vars=['Researcher'], var_name='Attribute', value_name='Value')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='Attribute', y='Value', hue='Researcher', marker='o')
    plt.title('Researcher Activities - Line Chart')
    plt.xlabel('Attributes')
    plt.ylabel('Values')
    plt.legend(title='Researcher')
    plt.show()

def scatter_plot_matrix(df):
    sns.pairplot(df, hue='Researcher', diag_kind='kde', palette='viridis')
    plt.suptitle('Researcher Activities - Scatter Plot Matrix', y=1.02)
    plt.show()

def bar_chart_with_error_bars(df):
    df_melted = pd.melt(df, id_vars=['Researcher'], var_name='Attribute', value_name='Value')
    df_grouped = df_melted.groupby('Attribute').agg({'Value': ['mean', 'std']}).reset_index()
    df_grouped.columns = ['Attribute', 'Mean', 'Std']

    plt.figure(figsize=(10, 6))
    plt.bar(df_grouped['Attribute'], df_grouped['Mean'], yerr=df_grouped['Std'], capsize=5, color='skyblue')
    plt.title('Researcher Activities - Bar Chart with Error Bars')
    plt.xlabel('Attributes')
    plt.ylabel('Mean Values')
    plt.show()

def violin_plot(df):
    df_melted = pd.melt(df, id_vars=['Researcher'], var_name='Attribute', value_name='Value')
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Attribute', y='Value', hue='Researcher', data=df_melted, split=True, palette='viridis')
    plt.title('Researcher Activities - Violin Plot')
    plt.xlabel('Attributes')
    plt.ylabel('Values')
    plt.show()
    
violin_plot(df)
""" line_chart(df)
scatter_plot_matrix(df)
bar_chart_with_error_bars(df)
 """