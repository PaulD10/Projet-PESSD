import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import ast
import gc
from tqdm.auto import tqdm
import re
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LargeInseeAnalyzer:
    def __init__(self, batch_size=1000, max_words=50000, min_word_freq=3):
        """Initialize analyzer with French-specific settings."""
        print("Initializing analyzer...")
        self.nlp = spacy.load('fr_core_news_md')
        self.batch_size = batch_size
        self.max_words = max_words
        self.min_word_freq = min_word_freq

        # Enhanced French stop words
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.stop_words.update([
            'insee', 'figure', 'encadré', 'être', 'avoir', 'faire',
            'dont', 'cet', 'cette', 'ces', 'celui', 'celle',
            'ainsi', 'alors', 'donc', 'etc', 'cas','onglet', 'ouvrir'
        ])

    def clean_text(self, text):
        """Clean individual text segment."""
        # Remove references
        text = re.sub(r'\[.*?\]', '', text)
        # Remove figure references
        text = re.sub(r'\(figure \d+\)', '', text)
        # Remove encadré references
        text = re.sub(r'\(encadré \d+\)', '', text)
        # Clean spaces
        text = ' '.join(text.split())
        return text

    def process_article(self, article_text):
        """Process a complete article (list of paragraphs)."""
        try:
            # Parse the string representation of list
            paragraphs = ast.literal_eval(article_text)

            # Process each paragraph
            processed_paragraphs = []
            for paragraph in paragraphs:
                # Clean the text
                cleaned_text = self.clean_text(paragraph)
                # Process with spaCy
                doc = self.nlp(cleaned_text.lower())

                # Extract relevant tokens
                tokens = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and
                        not token.is_stop and
                        not token.is_punct and
                        len(token.text) > 2 and  # Increased minimum length
                        token.lemma_ not in self.stop_words):

                        # Handle compound words
                        if '_' in token.text:
                            tokens.append(token.text.replace('_', ''))
                        else:
                            tokens.append(token.lemma_)

                if tokens:
                    processed_paragraphs.extend(tokens)

            return processed_paragraphs

        except Exception as e:
            print(f"Error processing article: {str(e)[:200]}")
            return []

    def process_in_batches(self, csv_path):
        """Process CSV file with article-level processing."""
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV file with {len(df)} articles")

            all_processed_texts = []
            total_articles = 0

            # Process each article
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
                try:
                    processed_tokens = self.process_article(row['full_text_sections'])
                    if processed_tokens:
                        all_processed_texts.append(processed_tokens)
                        total_articles += 1
                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)[:200]}")
                    continue

            print(f"\nSuccessfully processed {total_articles} articles")

            # Debug information
            if total_articles > 0:
                print("\nSample of processed tokens from first article:")
                print(all_processed_texts[0][:20])

            return all_processed_texts

        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            return []

    def create_focused_embeddings(self, processed_texts):
        """Create word embeddings with enhanced parameters."""
        if not processed_texts:
            raise ValueError("No processed texts available for embedding creation")

        print("\nAnalyzing word frequencies...")

        # Count word frequencies
        word_freq = defaultdict(int)
        for text in tqdm(processed_texts, desc="Counting words"):
            for token in text:
                word_freq[token] += 1

        # Filter by frequency
        filtered_freq = {word: freq for word, freq in word_freq.items()
                        if freq >= self.min_word_freq}

        print(f"\nFound {len(filtered_freq)} words occurring at least {self.min_word_freq} times")

        # Select top words
        top_words = dict(sorted(filtered_freq.items(),
                              key=lambda x: x[1],
                              reverse=True)[:self.max_words])

        print("\nTop 20 most frequent words:")
        for word, freq in list(top_words.items())[:20]:
            print(f"{word}: {freq}")

        # Prepare texts for Word2Vec
        filtered_texts = [[word for word in text if word in top_words]
                         for text in processed_texts]
        filtered_texts = [text for text in filtered_texts if text]

        # Train Word2Vec model
        print("\nTraining word embeddings...")
        model = Word2Vec(sentences=filtered_texts,
                        vector_size=200,
                        window=10,  # Increased context window
                        min_count=self.min_word_freq,
                        workers=4,
                        sg=1)  # Using skip-gram

        return model, filtered_freq

def analyze_insee_articles(csv_path, batch_size=1000, max_words=50000, min_word_freq=3):
    """Main analysis function."""
    print("\n=== Starting INSEE Article Analysis ===")
    print(f"Batch size: {batch_size}")
    print(f"Max words: {max_words}")
    print(f"Minimum word frequency: {min_word_freq}")

    analyzer = LargeInseeAnalyzer(
        batch_size=batch_size,
        max_words=max_words,
        min_word_freq=min_word_freq
    )

    processed_texts = analyzer.process_in_batches(csv_path)
    if not processed_texts:
        print("No texts were processed successfully")
        return None, None, None

    model, word_freq = analyzer.create_focused_embeddings(processed_texts)
    return model, None, word_freq

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict

class InseeClusterAnalyzer:
    def __init__(self, model, word_freq, n_clusters=12):
        self.model = model
        self.word_freq = word_freq
        self.n_clusters = n_clusters

    def prepare_vectors(self, min_freq=10):
        """Prepare word vectors for clustering."""
        print("Preparing vectors for clustering...")

        # Filter words by frequency
        frequent_words = {word: freq for word, freq in self.word_freq.items()
                        if freq >= min_freq and word in self.model.wv}

        # Get vectors
        words = list(frequent_words.keys())
        vectors = np.array([self.model.wv[word] for word in words])
        frequencies = [frequent_words[word] for word in words]

        return vectors, words, frequencies

    def reduce_dimensions(self, vectors):
        """Reduce dimensions using multiple methods."""
        print("\nReducing dimensions...")

        # UMAP reduction
        print("Performing UMAP reduction...")
        umap_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        umap_embedding = umap_reducer.fit_transform(vectors)

        # t-SNE reduction
        print("Performing t-SNE reduction...")
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            metric='cosine',
            n_iter=1000,
            random_state=42
        )
        tsne_embedding = tsne.fit_transform(vectors)

        return umap_embedding, tsne_embedding
    def analyze_clusters(self, labels, words, frequencies):
        """Analyze cluster composition and statistics."""
        # Créer un DataFrame avec les informations des clusters
        cluster_data = pd.DataFrame({
            'word': words,
            'frequency': frequencies,
            'cluster': labels
        })

        # Calculer les statistiques par cluster
        cluster_stats = []
        for cluster in sorted(cluster_data['cluster'].unique()):
            cluster_words = cluster_data[cluster_data['cluster'] == cluster]
            cluster_stats.append({
                'Cluster': cluster,
                'Size': len(cluster_words),
                'Average Frequency': cluster_words['frequency'].mean(),
                'Top Words': ', '.join(cluster_words.nlargest(5, 'frequency')['word'].tolist())
            })

        return pd.DataFrame(cluster_stats)

    def evaluate_clusters(self, vectors, min_clusters=2, max_clusters=30):
        """Évalue différents nombres de clusters avec plusieurs métriques."""
        print("\nÉvaluation du nombre optimal de clusters...")

        n_clusters_range = range(min_clusters, max_clusters + 1)
        metrics = {
            'silhouette': [],
            'calinski': [],
            'davies': [],
            'inertia': []
        }

        for n in tqdm(n_clusters_range, desc="Évaluation des clusters"):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)

            metrics['silhouette'].append(silhouette_score(vectors, labels, metric='cosine'))
            metrics['calinski'].append(calinski_harabasz_score(vectors, labels))
            metrics['davies'].append(davies_bouldin_score(vectors, labels))
            metrics['inertia'].append(kmeans.inertia_)

        return metrics, n_clusters_range

    def plot_optimization_metrics(self, metrics, n_clusters_range):
        """Visualise les métriques d'optimisation."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Score Silhouette (↑)',
                'Score Calinski-Harabasz (↑)',
                'Score Davies-Bouldin (↓)',
                'Inertie (Méthode du coude) (↓)'
            )
        )

        fig.add_trace(
            go.Scatter(x=list(n_clusters_range), y=metrics['silhouette'],
                      mode='lines+markers', name='Silhouette'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=list(n_clusters_range), y=metrics['calinski'],
                      mode='lines+markers', name='Calinski-Harabasz'),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=list(n_clusters_range), y=metrics['davies'],
                      mode='lines+markers', name='Davies-Bouldin'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=list(n_clusters_range), y=metrics['inertia'],
                      mode='lines+markers', name='Inertie'),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Métriques d'évaluation des clusters",
            showlegend=True
        )

        fig.show()

    def find_optimal_clusters(self, vectors, min_clusters=2, max_clusters=30):
        """Détermine le nombre optimal de clusters."""
        metrics, n_clusters_range = self.evaluate_clusters(vectors, min_clusters, max_clusters)

        # Visualisation des métriques
        self.plot_optimization_metrics(metrics, n_clusters_range)

        # Normalisation des métriques
        norm_silhouette = np.array(metrics['silhouette'])
        norm_calinski = (np.array(metrics['calinski']) - min(metrics['calinski'])) / \
                       (max(metrics['calinski']) - min(metrics['calinski']))
        norm_davies = 1 - (np.array(metrics['davies']) - min(metrics['davies'])) / \
                     (max(metrics['davies']) - min(metrics['davies']))
        norm_inertia = 1 - (np.array(metrics['inertia']) - min(metrics['inertia'])) / \
                      (max(metrics['inertia']) - min(metrics['inertia']))

        # Score composite
        composite_score = (norm_silhouette + norm_calinski + norm_davies + norm_inertia) / 4

        # Trouver le meilleur nombre de clusters
        optimal_idx = np.argmax(composite_score)
        optimal_n = list(n_clusters_range)[optimal_idx]

        best_by_metric = {
            'silhouette_best': list(n_clusters_range)[np.argmax(metrics['silhouette'])],
            'calinski_best': list(n_clusters_range)[np.argmax(metrics['calinski'])],
            'davies_best': list(n_clusters_range)[np.argmin(metrics['davies'])],
            'composite_best': optimal_n
        }

        print("\nNombre optimal de clusters par métrique:")
        for metric, value in best_by_metric.items():
            print(f"{metric}: {value}")

        return optimal_n, metrics, n_clusters_range

    def perform_clustering(self, vectors):
        """Perform clustering using multiple algorithms."""
        print("\nPerforming clustering...")

        # K-Means clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        kmeans_labels = kmeans.fit_predict(vectors)

        # DBSCAN clustering
        dbscan = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='cosine'
        )
        dbscan_labels = dbscan.fit_predict(vectors)

        return kmeans_labels, dbscan_labels

    def visualize_clusters_plotly(self, embedding, labels, words, frequencies,
                                method_name):
        """Create interactive cluster visualization with Plotly."""

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'word': words,
            'frequency': frequencies,
            'cluster': labels
        })

        # Calculate frequency percentile for each word
        df['freq_percentile'] = df['frequency'].rank(pct=True)

        # Determine which words to label (top 10% by frequency)
        df['show_label'] = df['freq_percentile'] > 0.9

        # Create the scatter plot
        fig = go.Figure()

        # Add scatter points for each cluster
        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]

            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers+text',
                    marker=dict(
                        size=np.sqrt(cluster_data['frequency']) * 2,
                        opacity=0.7
                    ),
                    text=cluster_data.apply(
                        lambda x: x['word'] if x['show_label'] else '',
                        axis=1
                    ),
                    textposition="top center",
                    name=f'Cluster {cluster}',
                    hovertext=cluster_data.apply(
                        lambda x: f"Mot: {x['word']}<br>Fréquence: {x['frequency']}<br>Cluster: {x['cluster']}",
                        axis=1
                    ),
                    hoverinfo='text'
                )
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Clusters de Mots ({method_name})',
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=True,
            legend_title="Clusters",
            hovermode='closest',
            plot_bgcolor='white',
            width=1200,
            height=800,
        )

        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        # Show the plot
        fig.show()

    def create_cluster_summary_plot(self, cluster_analysis):
        """Create interactive summary visualization of clusters."""

        # Prepare data for visualization
        clusters = cluster_analysis['Cluster'].tolist()
        sizes = cluster_analysis['Size'].tolist()
        avg_freqs = cluster_analysis['Average Frequency'].tolist()

        # Create the figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Taille des Clusters', 'Fréquence Moyenne par Cluster'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Add size bar chart
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {c}' for c in clusters],
                y=sizes,
                name='Taille',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # Add frequency bar chart
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {c}' for c in clusters],
                y=avg_freqs,
                name='Fréquence Moyenne',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Analyse des Clusters",
            showlegend=False,
            height=500,
            width=1200
        )

        fig.show()


    def perform_analysis(self, min_freq=10, optimize_clusters=True, min_clusters=2, max_clusters=30):
        """Perform complete clustering analysis with optional optimization."""
        # Prepare data
        vectors, words, frequencies = self.prepare_vectors(min_freq)

        if optimize_clusters:
            print("\nOptimisation du nombre de clusters...")
            optimal_n, _, _ = self.find_optimal_clusters(vectors, min_clusters, max_clusters)
            self.n_clusters = optimal_n
            print(f"\nUtilisation du nombre optimal de clusters: {optimal_n}")

        # Reduce dimensions
        umap_embedding, tsne_embedding = self.reduce_dimensions(vectors)

        # Perform clustering
        kmeans_labels, dbscan_labels = self.perform_clustering(vectors)

        # Visualize using both methods
        print("\nGenerating visualizations...")
        self.visualize_clusters_plotly(umap_embedding, kmeans_labels, words,
                                     frequencies, "UMAP")
        self.visualize_clusters_plotly(tsne_embedding, kmeans_labels, words,
                                     frequencies, "t-SNE")

        # Analyze clusters
        cluster_analysis = self.analyze_clusters(kmeans_labels, words, frequencies)

        # Show cluster summary
        print("\nGenerating cluster summary...")
        self.create_cluster_summary_plot(cluster_analysis)

        print("\nCluster Analysis:")
        print(cluster_analysis.to_string())
        export_clusters_to_txt(cluster_analysis, kmeans_labels, words, frequencies, "clusters_analysis.txt")
        # In your perform_analysis method, after clustering:


        return cluster_analysis

def analyze_clusters(model, word_freq, n_clusters=12, min_freq=10):
    """Main function for cluster analysis."""
    print("\n=== Starting Cluster Analysis ===")
    analyzer = InseeClusterAnalyzer(model, word_freq, n_clusters)
    return analyzer.perform_analysis(min_freq)

def export_clusters_to_txt(cluster_analysis, labels, words, frequencies, output_path="clusters.txt"):
    """
    Export clusters to a nicely formatted text file.

    Parameters:
    -----------
    cluster_analysis : pandas.DataFrame
        DataFrame containing cluster statistics
    labels : array-like
        Cluster labels for each word
    words : list
        List of words
    frequencies : list
        List of word frequencies
    output_path : str
        Path where to save the text file
    """
    # Create a dictionary to store words and frequencies for each cluster
    clusters = defaultdict(list)
    for word, freq, label in zip(words, frequencies, labels):
        clusters[label].append((word, freq))

    # Sort clusters by size (number of words)
    cluster_sizes = {k: len(v) for k, v in clusters.items()}
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("ANALYSE DES CLUSTERS DE MOTS\n")
        f.write("=" * 50 + "\n\n")

        # Write summary statistics
        f.write("STATISTIQUES GÉNÉRALES\n")
        f.write("-" * 30 + "\n")
        f.write(f"Nombre total de clusters: {len(clusters)}\n")
        f.write(f"Nombre total de mots: {sum(cluster_sizes.values())}\n")
        f.write(f"Moyenne de mots par cluster: {sum(cluster_sizes.values()) / len(clusters):.1f}\n")
        f.write("\n")

        # Write detailed cluster information
        f.write("DÉTAIL DES CLUSTERS\n")
        f.write("-" * 30 + "\n\n")

        for cluster_id, words_freq in sorted_clusters:
            # Get cluster statistics
            cluster_stats = cluster_analysis[cluster_analysis['Cluster'] == cluster_id].iloc[0]

            # Sort words by frequency
            sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)

            # Write cluster header
            f.write(f"Cluster {cluster_id}\n")
            f.write(f"{'-' * 20}\n")

            # Write cluster statistics
            f.write(f"Taille: {len(sorted_words)} mots\n")
            f.write(f"Fréquence moyenne: {cluster_stats['Average Frequency']:.1f}\n")

            # Write words and their frequencies
            f.write("\nMots les plus fréquents:\n")
            for word, freq in sorted_words:
                f.write(f"{word}: {freq}\n")

            f.write("\n" + "=" * 50 + "\n\n")

    print(f"Clusters exported to {output_path}")

    # Return a summary of the export
    return {
        'total_clusters': len(clusters),
        'total_words': sum(cluster_sizes.values()),
        'file_path': output_path
    }

# Put this at the end of your file, replacing the existing if __name__ == "__main__": block
if __name__ == "__main__":
    try:
        # Define the path to your CSV file

        # First step: Process articles and create embeddings
        print("\nStep 1: Processing articles and creating embeddings...")
        ls_periode = ["/home/paul/Desktop/INSEE projet stat/output/insee_publications_full.csv","/home/paul/Desktop/INSEE projet stat/output/data_2010_2014.csv","/home/paul/Desktop/INSEE projet stat/output/data_2015_2019.csv","/home/paul/Desktop/INSEE projet stat/output/data_2020_2024.csv"]
        for i in ls_periode:
            model, _, word_freq = analyze_insee_articles(
                csv_path=i,
                batch_size=1000,
                max_words=100000,
                min_word_freq=2
            )

            # Check if the first step was successful
            if model is None or word_freq is None:
                raise ValueError("Failed to create word embeddings. Check the input data and parameters.")

            # Second step: Perform clustering analysis
            print("\nStep 2: Performing clustering analysis...")
            cluster_analysis = analyze_clusters(
                model=model,
                word_freq=word_freq,
                n_clusters=15,
                min_freq=5
            )

            print("\nAnalysis completed successfully!")

        except FileNotFoundError:
            print(f"Error: Could not find the file at {csv_path}")
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
