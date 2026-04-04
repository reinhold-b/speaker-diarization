"""
Visu tool to visualize embeddings
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("data.csv", header=None).to_numpy()
    print(data.shape)

    scaler = StandardScaler().fit(data)
    data_scaled = scaler.transform(data)

    n_components = min(2, data_scaled.shape[0], data_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit_transform(data_scaled)
    print(pca.explained_variance_ratio_)

    data_pc = pca.transform(data_scaled)

    cluster = DBSCAN(eps=3, min_samples=1).fit(data_pc)
    labels = cluster.labels_

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        data_pc[:, 0],
        data_pc[:, 1],
        c=labels,
        cmap="tab10",
        s=40,
        alpha=0.85,
    )
    legend = ax.legend(*scatter.legend_elements(), loc="lower right", title="Cluster")
    ax.add_artist(legend)
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.show()
    pass

if __name__ == "__main__":
    main()