# SpaCy model for segmentation, tokenization, stopwords and stemming
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS

# Models for Latent Semantic Indexing
from gensim import corpora
from gensim import models
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel

class LSASummarizer:
	def __init__(self):
		self.nlp = spacy.load("fr_core_news_sm") # Model trained on French News

	def get_top_sentences(self, num_topics, top_scores, article_size):
		"""
			Picks the ordered list of indices of the best sentences to summarize the text
			Arguments:
				`num_topics`   The number of topics used in the LSI model
							   Example: 2
				`top_scores`   An array containing the top sentences (index and score) for each model
							   Example: [[(3, 0.5), (4, 0.35), (1, 0.15)], [(6, 0.75), (1, 0.45), (2, 0.3)]]
				`article_size` The number of sentences in the original article
			Returns:
				A list of the indices of the top sentences
		"""
		# Algorithm: First choose the best sentence of each topic
		# Then choose the second best sentence of each topic, then the third...
		# Keep going until the desired number of sentences has been reached
		top_sentences = []
		for i in range(article_size):
			for j in range(num_topics):
				if i >= len(top_scores[j]):
					continue
				if top_scores[j][i][0] not in top_sentences:
					top_sentences.append(top_scores[j][i][0])
		return top_sentences


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

		# Split the text into sentences, remove stopwords, stem words and remove punctuation
		sentences = []
		cur_sentence = []
		for sent in doc.sents:
			for token in sent:
				if not token.text.lower() in STOP_WORDS and not token.is_punct:
					cur_sentence.append(token.lemma_)
			sentences.append(cur_sentence)
			cur_sentence = []

		# Convert sentences to bags of words
		dictionary = corpora.Dictionary(sentences)
		doc_term_matrix = [dictionary.doc2bow(doc) for doc in sentences]

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
			coherencemodel = CoherenceModel(model=model, texts=sentences, dictionary=dictionary)
			coherence_values.append(coherencemodel.get_coherence())

		# Pick the number of topics that gives the highest coherence values
		max_coherence = coherence_values.index(max(coherence_values))
		num_topics = 2 + max_coherence
		model = model_list[max_coherence]

		# Apply the LSI model to our corpus
		corpus_lsi = model[doc_term_matrix]

		# Compute and store the scores of each sentence for each topic
		top_scores = [[] for i in range(num_topics)]
		for i, scores in enumerate(corpus_lsi):
			for j, score in scores:
				top_scores[j].append((i, abs(score)))

		# Sort the tables so that they contain the sentences in decreasing score order
		for topic in top_scores:
			topic.sort(reverse=True, key=lambda x: x[1])

		# Get a list of all sentences in decreasing order of importance
		sents = list(doc.sents)
		top_sentences = self.get_top_sentences(num_topics, top_scores, len(sents) + 1)

		# Try to add each sentence to the summary, starting from the best one
		# and making sure to not go over a tweet's length
		sents_to_add = []
		summary_size = 0
		for i in top_sentences:
			full_sent = sents[i].text
			new_size = summary_size + len(full_sent)
			if summary_size + new_size <= 280:
				sents_to_add.append(i)
				summary_size += len(full_sent) + 1 # +1 because of the space/newline between sentences

		# Now that we have the optimal list of sentences,
		# build the actual summary as well as the keyword-only version
		summary = ''
		keyword_summary = ''
		for sent_idx in sents_to_add:
			keyword_sent = ' '.join(word for word in sentences[sent_idx])
			full_sent = sents[sent_idx].text
			keyword_summary += keyword_sent + '\n'
			summary += full_sent + '\n'

		# Remove the final space/newline
		if summary:
			summary = summary[:-1]
			keyword_summary = keyword_summary[:-1]
		return summary, keyword_summary
