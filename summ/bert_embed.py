import string

from .utils import build_summary, get_keyword_sentences, get_top_sentences

# Various matrix/neural network modules
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch

# Cosine similarity (embeddings distance)
from scipy.spatial.distance import cosine

# SpaCy model for segmentation, tokenization and stopword removal
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS

# K-means clustering
from sklearn.cluster import KMeans

class BertEmbeddingsSummarizer:
    def __init__(self, model='flaubert/flaubert_large_cased'): # TODO camembert/camembert-large
        self.nlp = spacy.load("fr_core_news_sm") # Model trained on French News
        self.model_name = model
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def get_sent_embeds(self, tokenized_sentence):
        """
            Returns the embedding of words in a given sentence
            Arguments:
                `tokenized_sentence` The tokenized sentence whose words to embed
            Returns:
                The embeddings of the given sentence's word
        """
        
        encoded_sentence = self.tokenizer.encode(tokenized_sentence, is_split_into_words=True)
        
        if 'camembert' in self.model_name: # https://huggingface.co/camembert/camembert-large
            encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
            embeddings = self.model(encoded_sentence)[0]
        elif 'flaubert' in self.model_name: # https://huggingface.co/flaubert/flaubert_large_cased
            token_ids = torch.tensor([encoded_sentence])
            embeddings = self.model(token_ids)[0]
        else:
            raise ValueError('Unsupported model <{0}>'.format(self.model_name))
        return embeddings

    def get_summary(self, article, num_clusters=5):
        """
            Computes the optimal summary of an article using Bert embeddings and k-means clustering
            Arguments:
                `article`      The raw text content of the original article (without title)
                `num_clusters` The number of clusters to use in summarization
            Returns a tuple containing:
                - The generated summary in text form
                - A keywords-only version of the generated summary
        """

        doc = self.nlp(article)
        keyword_sentences = get_keyword_sentences(doc)
        
        embeddings = [] # List of embeddings for each keyword in the text
        word_idx_to_sent = [] # Maps indices in embeddings back to the original (sentence, word) indices
        
        # Get embeddings for all keywords in a sentence
        for i, sent in enumerate(doc.sents):
            tokenized_sent = self.tokenizer.tokenize(sent.text)
            # Get embeddings of all the words in the sentence
            sentence_embeds = self.get_sent_embeds(tokenized_sent)
            
            for j, token in enumerate(tokenized_sent):
                # Skip beginning- and end-of-sentence tokens
                if j == 0 or j == len(tokenized_sent):
                    continue
                
                # Keep only the embeddings keywords
                if not token in STOP_WORDS and token not in string.punctuation:
                    embeddings.append(sentence_embeds[0][j].detach().numpy())
                    word_idx_to_sent.append((i, j))

        # Stack all embeddings into a 2D matrix
        embeddings = np.stack(embeddings)
        
        # Perform k-means clustering on the embeddings
        kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)
        embed_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        # For each cluster, get all words associated with that cluster
        clusters = {}
        for cluster in range(num_clusters):
            indices = np.where(embed_labels == cluster)[0]
            # Then compute the score of each word in the cluster (distance to the centroid)
            for idx in indices:
                sent, word = word_idx_to_sent[idx]
                clusters[(sent, word)] = (cluster, cosine(embeddings[idx], centroids[cluster]))
        
        # Organize the scores by cluster
        top_scores = [[] for i in range(num_clusters)]
        for (sent, word), (cluster, score) in clusters.items():
            top_scores[cluster].append(((sent, word), score))

        # Pick the best summary using the computed scores
        return build_summary(top_scores, doc, keyword_sentences)
