from bert_score import BERTScorer

# to create keyword reference summary
from spacy.lang.fr import STOP_WORDS

from .eval import Eval

# BERTScore Implementation


class BERT_Eval(Eval):
    def __init__(self):
        self.scorer = BERTScorer(lang='fr', rescale_with_baseline=True)
        super().__init__()

    def split_summs(self, gen_summs, ref_summs, gen_keys=False):

        '''
        gen_summs = generated summaries list (containing pairs of long and keyword summaries
        ref_summs = reference summaries list (only containing long reference summaries)
        gen_keys = whether we want to create the list of keyword generated summaries and
                   generate the keyword reference summaries or not
        '''

        '''
        Function to separate long and keyword summaries for generated and reference data
        * I separated this from the actual evaluation so we could access all of these summaries to look at the matricies if we want
        using the get_matrix function lower down in this file *
        '''

        #nlp = spacy.load("fr_core_news_sm")

        # tried just unpacking the two variables from gen_summs but it wasn't working for me, I have no idea why
        # so I made a 'for loop' to do the same thing

        # If we want to generate keyword summaries
        if gen_keys == True:
            long_summs = []
            short_summs = []
            for x in range(len(gen_summs)):
                long_summs.append(gen_summs[x][0])
                short_summs.append(gen_summs[x][1])

            individual_summs = []
            cur_summ = []
            for summ in ref_summs:
                summ = self.nlp(summ)
                for sent in summ.sents:
                    for token in sent:
                        if not token.text.lower() in STOP_WORDS and not token.is_punct:
                            cur_summ.append(token.lemma_)
                individual_summs.append(cur_summ)
                cur_summ = []

            key_ref_summs = [' '.join(sent) for sent in individual_summs]

            return long_summs, short_summs, ref_summs, key_ref_summs

        # If we do not want to generate keyword summaries
        else:
            long_summs = []
            for x in range(len(gen_summs)):
                long_summs.append(gen_summs[x][0])

            return long_summs, ref_summs


    def bert_score(self, long_summs, ref_summs, short_summs=None, key_ref_summs=None, index=None):

        '''
        Function to compute the bert_scores for all the data
        * Add parameter x to look at score for only one article in data
        '''

        # Condition if we do want to look at the keyword summaries (short_summs & key_ref_summs)
        if short_summs != None and key_ref_summs != None:
            # Condition to look at average score for whole dataset
            if index == None:
                P_long, R_long, F1_long = self.scorer.score(long_summs, ref_summs, verbose=True)
                P_key, R_key, F1_key = self.scorer.score(short_summs, key_ref_summs, verbose=True)
                # P = precision
                # R = recall
                # F1 = F1-score

                results = {}
                results["Long precision avg"] = ('%.4f' % (P_long.mean()))
                results["Long recall avg"] = ('%.4f' % (R_long.mean()))
                results["Long F1-score avg"] = ('%.4f' % (F1_long.mean()))
                results["Keyword precision avg"] = ('%.4f' % (P_key.mean()))
                results["Keyword recall avg"] = ('%.4f' % (R_key.mean()))
                results["Keyword F1-score avg"] = ('%.4f' % (F1_key.mean()))

            # Condition to look at one score
            else:
                long_summ = [long_summs[index]]
                ref_summ = [ref_summs[index]]
                short_summ = [short_summs[index]]
                key_ref_summ = [key_ref_summs[index]]

                P_long, R_long, F1_long = self.scorer.score(long_summ, ref_summ, verbose=True)
                P_key, R_key, F1_key = self.scorer.score(short_summ, key_ref_summ, verbose=True)

                results = {}
                results["Long precision avg"] = ('%.4f' % (P_long))
                results["Long recall avg"] = ('%.4f' % (R_long))
                results["Long F1-score avg"] = ('%.4f' % (F1_long))
                results["Keyword precision avg"] = ('%.4f' % (P_key))
                results["Keyword recall avg"] = ('%.4f' % (R_key))
                results["Keyword F1-score avg"] = ('%.4f' % (F1_key))

        # Condition if we only want to evaluate the long/non-tokenized versions of summaries
        else:
            # Condition to look at average long scores for whole dataset
            if index == None:
                P_long, R_long, F1_long = self.scorer.score(long_summs, ref_summs, verbose=True)

                results = {}
                results["Long precision avg"] = ('%.4f' % (P_long.mean()))
                results["Long recall avg"] = ('%.4f' % (R_long.mean()))
                results["Long F1-score avg"] = ('%.4f' % (F1_long.mean()))

            # Condition to look at one long score
            else:
                long_summ = [long_summs[index]]
                ref_summ = [ref_summs[index]]

                P_long, R_long, F1_long = self.scorer.score(long_summ, ref_summ, verbose=True)

                results = {}
                results["Long precision avg"] = ('%.4f' % (P_long))
                results["Long recall avg"] = ('%.4f' % (R_long))
                results["Long F1-score avg"] = ('%.4f' % (F1_long))

        return results


    def get_matrix(self, gen_summ, ref_summ, index):

        '''
        Function to look at the relation matrix between any two generated and reference summaries
        '''

        return self.scorer.plot_example(gen_summ[index], ref_summ[index])

'''
# Example of how functions could be implemented in main.ipynb
# just don't forget to add class name before each of the functions

- Calling the split_summs function and storing outputs into variables to be used in last two functions
long_summs, short_summs, ref_summs, key_ref_sums =  split_summs(gen_summs, ref_summs)

- Calling the bert_score function
bert_score(long_summs, short_summs, ref_summs, key_ref_sums)

- Calling the get_matrix function
get_matrix(long_summs, ref_summs, 4) 
'''
