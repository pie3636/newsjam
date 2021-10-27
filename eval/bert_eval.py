# BERTScore import
# you can install this package by running pip install bert-score
from bert_score import BERTScorer

# spacy import; to create keyword reference summary
import spacy
from spacy.lang.fr import STOP_WORDS

# BERTScore Implementation
# I plan on putting all of these into a class later, so we can easily use them in the main file
# for now, you can copy this code to the bottom of main.ipynb to see the outputs/test it on our data

# *Note* I've tried running the bert_score on one article and it does not seem to work, I think input needs to be a list of sentences

class BERT_Eval:
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_sm")
        pass

    def split_summs(self, gen_summs, ref_summs):

        '''
        Function to separate long and keyword summaries for generated and reference data
        * I separated this from the actual evaluation so we could access all of these summaries to look at the matricies if we want
        using the get_matrix function lower down in this file *
        '''

        #nlp = spacy.load("fr_core_news_sm")

        # tried just unpacking the two variables from gen_summs but it wasn't working for me, I have no idea why
        # so I made a for loop to do the same thing
        long_summs = []
        short_summs = []
        for x in range(len(gen_summs)):
            long_summs.append(gen_summs[x][0])
            short_summs.append(gen_summs[x][1])

        summ_sentences = []
        summ_cur_sentence = []
        for summ in ref_summs:
            summ = self.nlp(summ)
            for sent in summ.sents:
                for token in sent:
                    if not token.text.lower() in STOP_WORDS and not token.is_punct:
                        summ_cur_sentence.append(token.lemma_)
                summ_sentences.append(summ_cur_sentence)
                summ_cur_sentence = []

        key_ref_summs = [' '.join(sent) for sent in summ_sentences]

        return long_summs, short_summs, ref_summs, key_ref_summs


    def bert_score(self, long_summs, short_summs, ref_summs, key_ref_summs):

        '''
        Function to compute the bert_scores for all the data
        * At this point I have the function working for the long summaries, but the keyword summary computation
        gives me an error relating to a mismatch of tensor sizes (need to look into further)
        '''

        # Instantiation of BERTScore
        scorer = BERTScorer(lang='fr', rescale_with_baseline=True)

        P_long, R_long, F1_long = scorer.score(long_summs, ref_summs, verbose=True)

        # Commented out because keyword summary evaluation is not currently working
        #P_key, R_key, F1_key = scorer.score(short_summs, key_ref_summs, verbose=True)
        # P = precision
        # R = recall
        # F1 = F1-score

        results = {}
        results["Long precision avg"] = P_long.mean()
        results["Long recall avg"] = R_long.mean()
        results["Long F1-score avg"] = F1_long.mean()

        # Commented out because keyword summary evaluation is not currently working
        #results["Keyword precision avg"] = P_key.mean()
        #results["Keyword recall avg"] = R_key.mean()
        #results["Keyword F1-score avg"] = F1_key.mean()

        return results


    def get_matrix(self, gen_summ, ref_summ, index):

        '''
        Function to look at the relation matrix between any two generated and reference summaries
        '''

        scorer = BERTScorer(lang='fr', rescale_with_baseline=True)
        matrix = scorer.plot_example(gen_summ[index], ref_summ[index])

        return matrix


# Example of how functions could be implemented in main.ipynb
# Calling the split_summs function and storing outputs into variables to be used in last two functions
#long_summs, short_summs, ref_summs, key_ref_sums =  split_summs(gen_summs, ref_summs)

# Calling the bert_score function
#bert_score(long_summs, short_summs, ref_summs, key_ref_sums)

# Calling the get_matrix function
#get_matrix(long_summs[0], ref_summs[0])