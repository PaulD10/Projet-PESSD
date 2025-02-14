import pandas as pd
import numpy as np
import spacy
import re
import umap
import hdbscan
import plotly.graph_objects as go
from tqdm.auto import tqdm
from collections import defaultdict
from gensim.models import Word2Vec, FastText
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gap_statistic import OptimalK

class INSEETextAnalyzer:
    def __init__(self, min_word_freq=2, max_words=500000):
        print("Initializing text analyzer...")
        self.nlp = spacy.load('fr_core_news_md', disable=["ner", "parser"])
        self.min_word_freq = min_word_freq
        self.max_words = max_words
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.stop_words.update(["insee", "figure", "être", "avoir", "ainsi", "dont", "etc"])
        self.processed_texts = []

    def clean_text(self, text):
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(figure \d+\)', '', text)
        text = ' '.join(text.split()).lower()
        return text

    def process_texts(self, texts):
        processed_texts = []
        for text in tqdm(texts, desc="Processing texts"):
            doc = self.nlp(self.clean_text(text))
            tokens = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]
                      and token.lemma_ not in self.stop_words and len(token.text) > 2]
            if tokens:
                processed_texts.append(tokens)
        return processed_texts

    def train_word_embeddings(self, processed_texts, model_type='word2vec'):
        print(f"Training {model_type} model...")
        model_class = Word2Vec if model_type == 'word2vec' else FastText
        model = model_class(sentences=processed_texts, vector_size=200, window=10, min_count=self.min_word_freq, sg=1, workers=4)
        return model

class INSEEClusterAnalyzer:
    def __init__(self, model, processed_texts):
        self.model = model
        self.processed_texts = processed_texts

    def get_word_vectors(self):
        words = list(self.model.wv.index_to_key)
        vectors = np.array([self.model.wv[word] for word in words])
        return words, vectors

    def reduce_dimensions(self, vectors):
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, metric='cosine')
        return reducer.fit_transform(vectors)

    def determine_optimal_clusters(self, vectors):
        optimal_k = OptimalK(parallel_backend='joblib')
        n_clusters = optimal_k(vectors, cluster_array=np.arange(2, 20))
        print(f"Optimal n_clusters determined: {n_clusters}")
        return n_clusters

    def perform_clustering(self, vectors, method='kmeans'):
        if method == 'kmeans':
            n_clusters = self.determine_optimal_clusters(vectors)
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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

    def visualize_clusters(self, embedding, labels, words, method):
        df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "word": words, "cluster": labels})
        fig = go.Figure()
        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            fig.add_trace(go.Scatter(x=cluster_data['x'], y=cluster_data['y'], mode='markers', name=f'Cluster {cluster}'))
        fig.update_layout(title_text=f"Visualization of {method} Clusters", title_x=0.5)
        fig.show()

if __name__ == "__main__":
    df = pd.read_csv("/home/paul/Desktop/INSEE projet stat/output/insee__premiere_publications_full.csv")
    texts = df['full_text_sections'].dropna().tolist()

    analyzer = INSEETextAnalyzer()
    processed_texts = analyzer.process_texts(texts)

    model = analyzer.train_word_embeddings(processed_texts, model_type='word2vec')

    cluster_analyzer = INSEEClusterAnalyzer(model, processed_texts)
    words, vectors = cluster_analyzer.get_word_vectors()
    reduced_vectors = cluster_analyzer.reduce_dimensions(vectors)

    for method in ['kmeans', 'agglomerative']:
        labels = cluster_analyzer.perform_clustering(reduced_vectors, method=method)
        cluster_analyzer.export_clusters(words, labels, method)
        cluster_analyzer.visualize_clusters(reduced_vectors, labels, words, method)

    print("Analysis complete.")
