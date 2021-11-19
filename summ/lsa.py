from .utils import build_summary, get_keyword_sentences, get_top_sentences

# Models for Latent Semantic Indexing
from gensim import corpora
from gensim import models
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel

from summarizer import Summarizer

class LSASummarizer(Summarizer):
    def __init__(self, max_len=280):
        super().__init__(max_len)


    def get_summary(self, article):
        """
            Computes the optimal summary of an article using Latent Semantic Analysis
            Arguments:
                `article` The raw text content of the original article (without title)
            Returns a tuple containing:
                - The generated summary in text form
                - A keywords-only version of the generated summary
        """

        doc = self.nlp(article)
        keyword_sentences = get_keyword_sentences(doc)

        # Convert sentences to bags of words
        dictionary = corpora.Dictionary(keyword_sentences)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in keyword_sentences]

        # Create a TF-IDF model that gives each word in each sentence a frequency score
        tfidf = models.TfidfModel(doc_term_matrix)
        sentences_tfidf = tfidf[doc_term_matrix]

        # Try to find the optimal number of topics for Latent Semantic Indexing
        # For that, we try using 2, 3, ..., 10 topics and we compute the coherence values
        # of the model for each number of topics.
        coherence_values = []
        model_list = []
        for num_topics in range(2, 10):
            model = LsiModel(sentences_tfidf, num_topics=num_topics, id2word=dictionary)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=keyword_sentences, dictionary=dictionary)
            coherence_values.append(coherencemodel.get_coherence())

        # Pick the number of topics that gives the highest coherence values
        max_coherence = coherence_values.index(max(coherence_values))
        num_topics = 2 + max_coherence
        model = model_list[max_coherence]

        # Apply the LSI model to our corpus
        corpus_lsi = model[doc_term_matrix]

        # Organize the scores by topic
        top_scores = [[] for i in range(num_topics)]
        for i, scores in enumerate(corpus_lsi):
            for j, score in scores:
                top_scores[j].append((i, abs(score)))

        # Pick the best summary using the computed scores
        return build_summary(top_scores, doc, keyword_sentences, self.max_len)
