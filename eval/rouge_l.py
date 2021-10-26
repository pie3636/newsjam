import spacy
from spacy.lang.fr.stop_words import STOP_WORDS

# Evaluation
from rouge_score import rouge_scorer
from tqdm import tqdm

class RougeLEval:
	def __init__(self):
		self.nlp = spacy.load("fr_core_news_sm") # Model trained on French News
		pass


	def evaluate_one(self, ref_summ, gen_summ):
		"""
			Computes the ROUGE-L score corresponding to the evaluation of a generated summary
			(in two versions: full text and keyword-only version) with a reference one
			Arguments:
				`ref_summ`	 The reference summary of the article
				`gen_summ`	 The generated summaries (full text and keywords-only version)
			Returns a tuple containing:
				- The scores of the full text summary
				- The scores of the keyword-only summary
		"""

		# Process the reference summary (segment it)
		# Also make a copy that is stemmed and has no stopwords to compare it with the
		# keyword-only generated summary
		long_summ, short_summ = gen_summ

		summ = self.nlp(ref_summ)
		summ_sentences = []
		summ_cur_sentence = []
		for sent in summ.sents:
			for token in sent:
				if not token.text.lower() in STOP_WORDS and not token.is_punct:
					summ_cur_sentence.append(token.lemma_)
			summ_sentences.append(summ_cur_sentence)
			summ_cur_sentence = []

		# Creates the instance that allows us to evaluate our summaries
		scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

		# Put the summaries together using newlines (required by RougeScorer)
		ref_summary = '\n'.join([sent.text for sent in summ.sents])
		keyword_ref_summary = '\n'.join([' '.join(sent) for sent in summ_sentences])

		# Compute and return the scores
		scores = scorer.score(ref_summary, long_summ)
		scores_keyword = scorer.score(keyword_ref_summary, short_summ)
		return scores, scores_keyword


	def evaluate_many(self, ref_summs, gen_summs, num_articles=None):
		"""
			Evaluates the summarization process for all articles in a set
			Arguments:
				`ref_summs`	A list containing the reference summaries of each article
				`gen_summs`	A list containing the generated summaries of each article
				`num_articles` The number of articles to evaluate (default: all)
			Returns a tuple containing:
				- The evaluation scores for all full generated summaries
				- The evaluation scores for all keywords-only generated summaries
		"""
		if num_articles is None:
			num_articles = len(ref_summs)

		long_eval_list = []
		keyword_eval_list = []

		for x in tqdm(range(num_articles)):
			long_eval, keyword_eval = self.evaluate_one(ref_summs[x], gen_summs[x])
			long_eval_list.append(long_eval)
			keyword_eval_list.append(keyword_eval)
		return long_eval_list, keyword_eval_list


	def get_results(self, long_scores, keyword_scores):
		"""
			Computes the average evaluation scores from a list
			Arguments:
				`long_scores`		 A list containing all evaluation scores for full generated summaries
				`keyword_scores` A list containing all evaluation scores for keyword-only generated summaries
			Returns a dict containing the average precision, recall and f1-score
				for both full and keyword_only generated summaries
		"""
		data_len = len(long_scores)

		results = {}
		results["Long precision avg"] = sum(map(lambda x: x['rougeL'][0], long_scores)) / data_len
		results["Long recall avg"] = sum(map(lambda x: x['rougeL'][1], long_scores)) / data_len
		results["Long F1-score avg"] = sum(map(lambda x: x['rougeL'][2], long_scores)) / data_len
		results["Keyword precision avg"] = sum(map(lambda x: x['rougeL'][0], keyword_scores)) / data_len
		results["Keyword recall avg"] = sum(map(lambda x: x['rougeL'][1], keyword_scores)) / data_len
		results["Keyword F1-score avg"] = sum(map(lambda x: x['rougeL'][2], keyword_scores)) / data_len

		return results
