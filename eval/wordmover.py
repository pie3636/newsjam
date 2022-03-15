from spacy.lang.fr.stop_words import STOP_WORDS

# Evaluation
import gensim
from gensim.models import fasttext, keyedvectors
from tqdm import tqdm

from .eval import Eval

class WordMoverEval(Eval):
    def __init__(self, in_file=None, model=None, pretraining={}):
        if model is not None:
            self.model = model
        elif in_file is not None:
            if in_file.endswith('.bin'):
                self.model = fasttext.load_model(in_file)
            else:
                self.model = keyedvectors.load_word2vec_format(in_file)
        else:
            raise ValueError('A fasttext model or input file must be specified.')
        if len(pretraining):
            sents = pretraining['sents']
            self.model.build_vocab(sents, update=True)
            self.model.train(sentences=sents, total_examples=len(sent), epochs=pretraining['epochs'])
        super().__init__()


    def evaluate_one(self, ref_summ, gen_summ):
        """
                Computes the word mover distance corresponding to the evaluation of a generated summary
                (in two versions: full text and keyword-only version) with a reference one
                Arguments:
                        `ref_summ`       The reference summary of the article
                        `gen_summ`       The generated summaries (full text and keywords-only version)
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

        # Put the summaries together using newlines (required by RougeScorer)
        ref_summary = '\n'.join([sent.text for sent in summ.sents])
        keyword_ref_summary = '\n'.join([' '.join(sent) for sent in summ_sentences])

        # Compute and return the scores
        scores = self.model.wmdistance(ref_summary, long_summ)
        scores_keyword = self.model.wv.wmdistance(keyword_ref_summary, short_summ)
        return scores, scores_keyword


    def evaluate_many(self, ref_summs, gen_summs, num_articles=None):
        """
                Evaluates the summarization process for all articles in a set
                Arguments:
                        `ref_summs`     A list containing the reference summaries of each article
                        `gen_summs`     A list containing the generated summaries of each article
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
				`keyword_scores`     A list containing all evaluation scores for keyword-only generated summaries
			Returns a dict containing the average score for both full and keyword_only generated summaries
		"""
		data_len = len(long_scores)

		results = {}
		results["Long avg"] = sum(long_scores) / data_len
		results["Keyword avg"] = sum(keyword_scores) / data_len
		
		return results

