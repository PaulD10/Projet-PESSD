# -*- coding: utf-8 -*-
"""
INSEE Text Analysis with Word Embeddings, Clustering, and Topic Modeling
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec, LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import ast
import gc
from tqdm.auto import tqdm
import re
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import umap
import pyLDAvis
import pyLDAvis.gensim_models

class LargeInseeAnalyzer:
    def __init__(self, batch_size=1000, max_words=5000000, min_word_freq=3):
        """Initialize analyzer with French-specific settings."""
        print("Initializing analyzer...")
        self.nlp = spacy.load('fr_core_news_md')
        self.batch_size = batch_size
        self.max_words = max_words
        self.min_word_freq = min_word_freq
        self.processed_texts = None

        # Enhanced French stop words
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.stop_words.update([
            'insee', 'figure', 'encadré', 'être', 'avoir', 'faire',
            'dont', 'cet', 'cette', 'ces', 'celui', 'celle',
            'ainsi', 'alors', 'donc', 'etc', 'cas', 'onglet', 'ouvrir'
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
                        len(token.text) > 2 and
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
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV file with {len(df)} articles")

            all_processed_texts = []
            total_articles = 0

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
            self.processed_texts = all_processed_texts
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
                        window=10,
                        min_count=self.min_word_freq,
                        workers=4,
                        sg=1)

        return model, filtered_freq

class InseeClusterAnalyzer:
    def __init__(self, model, word_freq, processed_texts=None, n_clusters=12):
        """Initialize with word embeddings and optional processed texts."""
        self.model = model
        self.word_freq = word_freq
        self.n_clusters = n_clusters
        self.processed_texts = processed_texts
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

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

    def evaluate_clusters(self, vectors, min_clusters=2, max_clusters=30):
        """Evaluate different numbers of clusters with multiple metrics."""
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
        """Visualize cluster optimization metrics."""
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
        """Determine optimal number of clusters."""
        metrics, n_clusters_range = self.evaluate_clusters(vectors, min_clusters, max_clusters)

        # Visualize metrics
        self.plot_optimization_metrics(metrics, n_clusters_range)

        # Normalize metrics
        norm_silhouette = np.array(metrics['silhouette'])
        norm_calinski = (np.array(metrics['calinski']) - min(metrics['calinski'])) / \
                       (max(metrics['calinski']) - min(metrics['calinski']))
        norm_davies = 1 - (np.array(metrics['davies']) - min(metrics['davies'])) / \
                     (max(metrics['davies']) - min(metrics['davies']))
        norm_inertia = 1 - (np.array(metrics['inertia']) - min(metrics['inertia'])) / \
                      (max(metrics['inertia']) - min(metrics['inertia']))

        # Composite score
        composite_score = (norm_silhouette + norm_calinski + norm_davies + norm_inertia) / 4

        # Find optimal number of clusters
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

    def analyze_clusters(self, labels, words, frequencies):
        """Analyze cluster composition and statistics."""
        cluster_data = pd.DataFrame({
            'word': words,
            'frequency': frequencies,
            'cluster': labels
        })

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

    def visualize_clusters_plotly(self, embedding, labels, words, frequencies,
                                method_name):
        """Create interactive cluster visualization with Plotly."""
        df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'word': words,
            'frequency': frequencies,
            'cluster': labels
        })

        df['freq_percentile'] = df['frequency'].rank(pct=True)
        df['show_label'] = df['freq_percentile'] > 0.9

        fig = go.Figure()

        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]

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

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        fig.show()

    def create_cluster_summary_plot(self, cluster_analysis):
        """Create interactive summary visualization of clusters."""
        clusters = cluster_analysis['Cluster'].tolist()
        sizes = cluster_analysis['Size'].tolist()
        avg_freqs = cluster_analysis['Average Frequency'].tolist()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Taille des Clusters', 'Fréquence Moyenne par Cluster'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(
                x=[f'Cluster {c}' for c in clusters],
                y=sizes,
                name='Taille',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=[f'Cluster {c}' for c in clusters],
                y=avg_freqs,
                name='Fréquence Moyenne',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title_text="Analyse des Clusters",
            showlegend=False,
            height=500,
            width=1200
        )

        fig.show()

    def prepare_for_lda(self, processed_texts=None, min_freq=5):
        """Prepare texts for LDA analysis."""
        if processed_texts is None and self.processed_texts is None:
            raise ValueError("No processed texts available for LDA analysis")

        texts_to_use = processed_texts if processed_texts is not None else self.processed_texts
        print("\nPreparing texts for LDA analysis...")

        # Create dictionary
        self.dictionary = Dictionary(texts_to_use)

        # Filter out rare and common words
        self.dictionary.filter_extremes(no_below=min_freq, no_above=0.5)

        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in texts_to_use]

        return self.dictionary, self.corpus

    def evaluate_lda_models(self, corpus, dictionary, processed_texts,
                          min_topics=2, max_topics=30, step=1):
        """Evaluate LDA models with different numbers of topics."""
        print("\nEvaluating LDA models...")

        coherence_scores = []
        perplexity_scores = []
        n_topics_range = range(min_topics, max_topics + 1, step)

        for n_topics in tqdm(n_topics_range, desc="Evaluating topic numbers"):
            # Train LDA model
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=n_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                eta='auto'
            )

            # Calculate coherence score
            coherence_model = CoherenceModel(
                model=lda,
                texts=processed_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_scores.append(coherence_model.get_coherence())

            # Calculate perplexity
            perplexity_scores.append(lda.log_perplexity(corpus))

        return n_topics_range, coherence_scores, perplexity_scores

    def plot_lda_metrics(self, n_topics_range, coherence_scores, perplexity_scores):
        """Plot LDA evaluation metrics."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Coherence Score (↑)', 'Perplexity Score (↓)')
        )

        fig.add_trace(
            go.Scatter(
                x=list(n_topics_range),
                y=coherence_scores,
                mode='lines+markers',
                name='Coherence'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=list(n_topics_range),
                y=perplexity_scores,
                mode='lines+markers',
                name='Perplexity'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            title_text="Métriques d'évaluation LDA",
            showlegend=True
        )

        fig.show()

    def train_lda_model(self, n_topics):
        """Train the final LDA model."""
        if self.corpus is None or self.dictionary is None:
            raise ValueError("Corpus and dictionary must be prepared first using prepare_for_lda()")

        print(f"\nTraining LDA model with {n_topics} topics...")

        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=n_topics,
            random_state=42,
            passes=20,
            alpha='auto',
            eta='auto'
        )

        return self.lda_model

    def visualize_lda_topics(self):
        """Create interactive visualization of LDA topics."""
        if self.lda_model is None or self.corpus is None or self.dictionary is None:
            raise ValueError("LDA model, corpus, and dictionary must be prepared first")

        try:
            # Prepare visualization
            vis_data = pyLDAvis.gensim_models.prepare(
                self.lda_model,
                self.corpus,
                self.dictionary,
                sort_topics=False
            )

            # Save visualization to HTML
            pyLDAvis.save_html(vis_data, 'lda_visualization.html')
            print("\nLDA visualization saved to 'lda_visualization.html'")
        except Exception as e:
            print(f"Error creating LDA visualization: {str(e)}")

    def export_lda_topics(self, output_path="lda_topics.txt"):
        """Export LDA topics to a text file."""
        if self.lda_model is None:
            raise ValueError("LDA model must be trained first")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("ANALYSE DES TOPICS LDA\n")
                f.write("=" * 50 + "\n\n")

                # Write topic information
                for topic_id in range(self.lda_model.num_topics):
                    f.write(f"Topic {topic_id + 1}\n")
                    f.write("-" * 20 + "\n")

                    # Get top words for topic
                    words = self.lda_model.show_topic(topic_id, topn=20)

                    for word, prob in words:
                        f.write(f"{word}: {prob:.4f}\n")

                    f.write("\n")

            print(f"LDA topics exported to {output_path}")
        except Exception as e:
            print(f"Error exporting LDA topics: {str(e)}")

    def perform_lda_analysis(self, processed_texts=None, min_freq=5,
                           optimize_topics=True, min_topics=2, max_topics=30):
        """Perform complete LDA analysis."""
        try:
            # Use provided texts or stored texts
            texts_to_use = processed_texts if processed_texts is not None else self.processed_texts
            if texts_to_use is None:
                raise ValueError("No processed texts available for LDA analysis")

            # Prepare data for LDA
            self.prepare_for_lda(texts_to_use, min_freq)

            if optimize_topics:
                # Evaluate different numbers of topics
                n_topics_range, coherence_scores, perplexity_scores = self.evaluate_lda_models(
                    self.corpus, self.dictionary, texts_to_use,
                    min_topics, max_topics
                )

                # Plot evaluation metrics
                self.plot_lda_metrics(n_topics_range, coherence_scores, perplexity_scores)

                # Find optimal number of topics (based on coherence)
                optimal_topics = list(n_topics_range)[np.argmax(coherence_scores)]
                print(f"\nOptimal number of topics based on coherence: {optimal_topics}")
            else:
                optimal_topics = self.n_clusters

            # Train final model
            self.train_lda_model(optimal_topics)

            # Create visualizations
            self.visualize_lda_topics()

            # Export topics
            self.export_lda_topics()

            return self.lda_model

        except Exception as e:
            print(f"Error during LDA analysis: {str(e)}")
            return None

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

        # Export cluster analysis
        export_clusters_to_txt(cluster_analysis, kmeans_labels, words, frequencies, "clusters_analysis.txt")

        return cluster_analysis

def export_clusters_to_txt(cluster_analysis, labels, words, frequencies, output_path="clusters.txt"):
    """Export clusters to a nicely formatted text file."""
    try:
        # Create a dictionary to store words and frequencies for each cluster
        clusters = defaultdict(list)
        for word, freq, label in zip(words, frequencies, labels):
            clusters[label].append((word, freq))

        # Sort clusters by size
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
        return {
            'total_clusters': len(clusters),
            'total_words': sum(cluster_sizes.values()),
            'file_path': output_path
        }

    except Exception as e:
        print(f"Error exporting clusters: {str(e)}")
        return None

def analyze_insee_articles(csv_path, batch_size=1000, max_words=50000, min_word_freq=3):
    """Main analysis function for INSEE articles."""
    print("\n=== Starting INSEE Article Analysis ===")

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
    return model, processed_texts, word_freq

if __name__ == "__main__":
    try:
        # Define path to CSV file
        csv_path = "/home/paul/Desktop/INSEE projet stat/output/insee__premiere_publications_full.csv"  # Update with your actual path
        print(f"Starting analysis of file: {csv_path}")

        # Process articles and create embeddings
        print("\nStep 1: Processing articles and creating embeddings...")
        model, processed_texts, word_freq = analyze_insee_articles(
            csv_path=csv_path,
            batch_size=1000,
            max_words=100000,
            min_word_freq=2
        )

        if model is None or processed_texts is None or word_freq is None:
            raise ValueError("Failed to process articles and create embeddings")

        # Create analyzer with processed texts
        analyzer = InseeClusterAnalyzer(
            model=model,
            word_freq=word_freq,
            processed_texts=processed_texts
        )
# Perform clustering analysis
        print("\nPerforming clustering analysis...")
        cluster_analysis = analyzer.perform_analysis(
            min_freq=5,
            optimize_clusters=True,
            min_clusters=2,
            max_clusters=30
        )

        if cluster_analysis is None:
            print("Warning: Clustering analysis failed")
        else:
            print("\nClustering analysis completed successfully")

        # Perform LDA analysis
        print("\nPerforming LDA analysis...")
        lda_model = analyzer.perform_lda_analysis(
            processed_texts=processed_texts,
            min_freq=5,
            optimize_topics=True,
            min_topics=15,
            max_topics=30
        )

        if lda_model is None:
            print("Warning: LDA analysis failed")
        else:
            print("\nLDA analysis completed successfully")

        print("\nComplete analysis finished successfully!")

        # Memory cleanup
        gc.collect()

    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}")
        print("Please check the file path and try again.")
    except ValueError as ve:
        print(f"Value Error: {str(ve)}")
        print("Please check your input parameters and try again.")
    except Exception as e:
        print(f"Unexpected error during analysis: {str(e)}")
        print("Please check the error message and your input data.")
        raise  # Re-raise the exception for debugging
