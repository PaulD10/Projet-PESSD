from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec
import plotly.graph_objs as go
import umap.umap_ as umap
import plotly.express as px
import matplotlib
#from gap_stat import OptimalK
import io
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import random

seed_value = 1312
random.seed(seed_value)
np.random.seed(seed_value)
# Load the trained model
#model = Word2Vec.load("/home/paul/Desktop/INSEE projet stat/output/insee_publications_complete/word2vec_insee_premiere_complet.model")

class INSEEClusterAnalyzer:
    def __init__(self, model):
        self.model = model

    def get_word_vectors(self):
        words = list(self.model.wv.index_to_key)
        vectors = np.array([self.model.wv[word] for word in words])
        return words, vectors

    def reduce_dimensions(self, vectors):
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, metric='cosine', random_state=seed_value)
        return reducer.fit_transform(vectors)

    def determine_optimal_clusters(self, vectors):
        #optimal_k = OptimalK(parallel_backend='joblib')
        n_clusters = 30
        print(f"Optimal n_clusters determined: {n_clusters}")
        return n_clusters

    def perform_clustering(self, vectors, method='kmeans'):
        if method == 'kmeans':
            n_clusters = self.determine_optimal_clusters(vectors)
            model = KMeans(n_clusters=n_clusters, random_state=seed_value , n_init=10)
        else:
            n_clusters = self.determine_optimal_clusters(vectors)
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        return model.fit_predict(vectors)

    def export_clusters(self, words, labels, method):
        cluster_data = pd.DataFrame({"word": words, "cluster": labels})
        output_file = f"clusters_{method}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for cluster in sorted(cluster_data['cluster'].unique()):
                f.write(f"Cluster {cluster}\n")
                f.write("-" * 20 + "\n")
                cluster_words = cluster_data[cluster_data['cluster'] == cluster]['word'].tolist()
                cluster_size = len(cluster_words)
                f.write(f"Cluster size: {cluster_size}\n")
                f.write(", ".join(cluster_words) + "\n\n")
        print(f"Clusters saved to {output_file}")



    def visualize_clusters(self, embedding, labels, words, method, clusters_themes=None):
        df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "word": words, "cluster": labels})
        fig = go.Figure()

        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]

            # Utiliser le nom thématique du cluster s'il est disponible
            if clusters_themes and cluster in clusters_themes:
                cluster_name = clusters_themes[cluster]
            else:
                cluster_name = f'Cluster {cluster}'

            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=cluster_name,
                text=cluster_data['word'],  # Ajouter les mots comme info au survol
                hoverinfo='text'
            ))

        fig.update_layout(
            title_text=f"Visualization of {method} Clusters",
            title_x=0.5,
            legend_title="Clusters thématiques"
        )

        fig.show()
if __name__ == "__main__":
    # Load the trained model
    model = Word2Vec.load("/home/paul/Desktop/INSEE projet stat/output/insee_publications_complete/word2vec_insee_premiere_complet.model")

    cluster_analyzer = INSEEClusterAnalyzer(model)
    words, vectors = cluster_analyzer.get_word_vectors()
    reduced_vectors = cluster_analyzer.reduce_dimensions(vectors)
    clusters_themes = []
    for method in ['kmeans', 'agglomerative']:
        labels = cluster_analyzer.perform_clustering(reduced_vectors, method=method)
        cluster_analyzer.export_clusters(words, labels, method)
        cluster_analyzer.visualize_clusters(reduced_vectors, labels, words, method, clusters_themes)

    print("Analysis complete.")
