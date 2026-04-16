"""
Visu tool to visualize embeddings
"""

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.test.test_vad_output import play_segments 
from lib.vad_data_loader import VADDataLoader
from lib.const import *

segments_data = VADDataLoader.load_from_json(
    filepath_vad=TEST_VAD_PATH,
    path_audio=TEST_AUDIO_PATH,
)

def play_audio_for_datapoint(index: int):
    play_segments([segments_data[index]], sample_rate=16000)

def main():
    embedding_data = pd.read_csv("data.csv", header=None).to_numpy()

    scaler = StandardScaler().fit(embedding_data)
    data_scaled = scaler.transform(embedding_data)
    data_scaled = normalize(data_scaled)

    pca = PCA(n_components=20)
    data_pc = pca.fit_transform(data_scaled)

    # Clustering auf 20D-PCA (cosine ist für Embeddings meist besser)
    cluster = DBSCAN(eps=0.3, min_samples=8, metric="cosine").fit(data_pc)
    labels = cluster.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"clusters={n_clusters}, noise={n_noise}")

    pca_plot = PCA(n_components=2, random_state=42)
    data_pc = pca_plot.fit_transform(data_scaled)

    fig, ax = plt.subplots()
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", max(1, len(unique_labels)))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = "#9e9e9e" if lbl == -1 else cmap(i)
        name = "noise" if lbl == -1 else f"cluster {lbl}"
        ax.scatter(
            data_pc[mask, 0],
            data_pc[mask, 1],
            s=40,
            alpha=0.9,
            c=[color],
            label=f"{name} ({int(np.sum(mask))})",
            picker=True,
            pickradius=0.01,
        )


    def on_pick(event):
        idx = int(event.ind[0])
        print(idx)
        play_audio_for_datapoint(idx)

    fig.canvas.mpl_connect("pick_event", on_pick)

    ax.legend(loc="lower right", title="Cluster")
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.show()

if __name__ == "__main__":
    main()