import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

"""Dislaying the clustering fusion results according to the 3 parameters: n_results, minimum_cluster_size and majority_threshold"""

res = np.load('fusion_results_mobilenet.npy')

df = pd.DataFrame(res,
                  columns=["n_results", "min_cluster_size", "majority_threshold", "loss", "n_excluded", "n_clusters"])
df['loss_score'] = df.n_excluded / 25 + df.loss

# fig = px.scatter_3d(df, x="n_excluded", y="n_clusters", color='n_results',z='loss',
#                     hover_data=["n_results", "min_cluster_size", "majority_threshold",'loss','n_excluded','n_clusters'])

fig = px.scatter(df, x=df.n_results, y=df.loss_score, color=df.majority_threshold, size=df.min_cluster_size,
                 hover_data=["n_results", "min_cluster_size", "majority_threshold", 'loss', 'n_excluded', 'n_clusters'],
                 )

fig.update_layout(
    title={
        'text': "Mobilenet",
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
# fig.update_traces(marker_size=4)
fig.write_html("results_fusion.html")

fig.show()
