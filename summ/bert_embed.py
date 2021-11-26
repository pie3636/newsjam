from .summarizer import Summarizer
from .utils import build_summary, get_keyword_sentences, get_top_sentences

# Various matrix/neural network modules
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

# Cosine similarity (embeddings distance)
from scipy.spatial.distance import cosine

# SpaCy model for segmentation, tokenization and stopword removal
import spacy

# K-means clustering
from sklearn.cluster import KMeans


class BertEmbeddingsSummarizer(Summarizer):
    def __init__(self, model='flaubert/flaubert_large_cased', max_len=280):
        self.model_name = model
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        super().__init__(max_len)

    def get_sent_embeds(self, encoded_sentence):
        """
            Returns the embedding of words in a given sentence
            Arguments:
                `encoded_sentence` The encoded sentence whose words to embed
            Returns:
                The embeddings of the given sentence's word
        """

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
        word_idx_to_sent = dict # Maps word embeddings back to the original (sentence, word) indices

        # Get embeddings for all keywords in a sentence
        for i, sent in enumerate(doc.sents):
            tokenized_sent = self.tokenizer.tokenize(sent.text)
            # Skip empty sentences
            if not tokenized_sent:
                continue

            # Get embeddings of all the words in the sentence
            encoded_sent = self.tokenizer.encode(tokenized_sent)
            sentence_embeds = self.get_sent_embeds(encoded_sent)
            
            # Add mapping for each embedding to its original word
            for embed in torch.unbind(sentence_embeds)[0]:
                word_idx_to_sent[embed] == (i, j)
                
            embeddings.append(sentence_embeds[0].detach())

        # Stack all embeddings into a 2D matrix
        embeddings = np.stack(embeddings)

        # Perform k-means clustering on the embeddings
        kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)
        embed_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # For each cluster, get all words associated with that cluster
        clusters = {}
        for cluster in range(num_clusters):
            embeds = np.where(embed_labels == cluster)[0]
            # Then compute the score of each word in the cluster (distance to the centroid)
            for embed_idx in embeds:
                embed = embeddings[embed_idx]
                clusters[word_idx_to_sent[embed]] = (cluster, 1/cosine(embed, centroids[cluster]))

        # Organize the scores by cluster
        top_scores = [[] for i in range(num_clusters)]
        for (sent, word), (cluster, score) in clusters.items():
            top_scores[cluster].append(((sent, word), score))

        # Pick the best summary using the computed scores
        return build_summary(top_scores, doc, keyword_sentences, self.max_len)

        """
            Similar to `get_summary` but works on a batch of sentences at once.
            Arguments:
                `article`     The raw text content of the original article (without title)
                `batch_size`  The number of sentences to process per batch
            Returns a list containing the same values as the result of `get_summary`.
        """

        doc = self.nlp(article)
        keyword_sentences = get_keyword_sentences(doc)

        embeddings = [] # List of embeddings for each keyword in the text
        word_idx_to_sent = [] # Maps indices in embeddings back to the original (sentence, word) indices

        # Get embeddings for all keywords in a sentence
        all_summaries = []
        for i in range(0, len(doc.sents), batch_size):
            cur_batch = doc.sents[i:i+batch_size]
            batch_len = len(batch)
            
            tokenized_sents = list(filter(self.tokenizer.tokenize(sent.text) for sent in batch))
            sentence_embeds = self.get_sent_embeds(tokenized_sents)
            
            for i, sent in enumerate(tokenized_sents):
                for j, token in enumerate(sent):
                    word_idx_to_sent.append((i, j))

            # Stack all embeddings into a 3D matrix
            all_embeddings = pad_sequence(sentence_embeds, batch_first=True)

            for embeddings in all_embeddings:
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
                all_summaries.append(build_summary(top_scores, doc, keyword_sentences, self.max_len))
        
        return all_summaries
