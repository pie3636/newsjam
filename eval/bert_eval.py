from bert_score import BERTScorer

# to create keyword reference summary
from spacy.lang.fr import STOP_WORDS as fr_stop
from spacy.lang.en import STOP_WORDS as en_stop

from .eval import Eval

# BERTScore Implementation


class BERT_Eval(Eval):
    def __init__(self):

        '''
        - 1 = rescale_with_baseline=True
        - 2 = rescale_with_baseline=False
        '''
        self.scorer_1 = BERTScorer(lang='fr', rescale_with_baseline=True)
        self.scorer_2 = BERTScorer(lang='fr', rescale_with_baseline=False)

        self.scorer_en_1 = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en', rescale_with_baseline=True)
        self.scorer_en_2 = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en', rescale_with_baseline=False)
        super().__init__()

    def split_summs(self, gen_summs, ref_summs,gen_keys=False, lang='fr'):

        '''
        gen_summs = generated summaries list (containing pairs of long and keyword summaries
        ref_summs = reference summaries list (only containing long reference summaries)
        gen_keys = whether we want to create the list of keyword generated summaries and
                   generate the keyword reference summaries or not
        lang = language of the BERTScore model
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
        if lang =='fr':
            stop_words = fr_stop
            nlp = self.nlp
        elif lang == 'en':
            stop_words = en_stop
            nlp = self.nlp_en

        if gen_keys == True:
            long_summs = []
            short_summs = []
            for x in range(len(gen_summs)):
                long_summs.append(gen_summs[x][0])
                short_summs.append(gen_summs[x][1])

            individual_summs = []
            cur_summ = []
            for summ in ref_summs:
                summ = nlp(summ)
                for sent in summ.sents:
                    for token in sent:
                        if not token.text.lower() in stop_words and not token.is_punct:
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


    def bert_score(self, long_summs, ref_summs, short_summs=None, key_ref_summs=None, index=None, lang='fr'):

        '''
        Function to compute the bert_scores for all the data
        '''

        if lang == 'fr':
            scorer_1 = self.scorer_1
            scorer_2 = self.scorer_2
        elif lang == 'en':
            scorer_1 = self.scorer_en_1
            scorer_2 = self.scorer_en_2


        # Condition if we do want to look at the keyword summaries (short_summs & key_ref_summs)
        if short_summs != None and key_ref_summs != None:
            # Condition to look at average score for whole dataset
            if index == None:
                P_long_1, R_long_1, F1_long_1 = scorer_1.score(long_summs, ref_summs, verbose=True)
                P_key_1, R_key_1, F1_key_1 = scorer_1.score(short_summs, key_ref_summs, verbose=True)

                P_long_2, R_long_2, F1_long_2 = scorer_2.score(long_summs, ref_summs, verbose=True)
                P_key_2, R_key_2, F1_key_2 = scorer_2.score(short_summs, key_ref_summs, verbose=True)
                # P = precision
                # R = recall
                # F1 = F1-score

                results = {}
                results["Rescaled Long precision avg"] = ('%.4f' % (P_long_1.mean()))
                results["Rescaled Long recall avg"] = ('%.4f' % (R_long_1.mean()))
                results["Rescaled Long F1-score avg"] = ('%.4f' % (F1_long_1.mean()))
                results["Rescaled Keyword precision avg"] = ('%.4f' % (P_key_1.mean()))
                results["Rescaled Keyword recall avg"] = ('%.4f' % (R_key_1.mean()))
                results["Rescaled Keyword F1-score avg"] = ('%.4f' % (F1_key_1.mean()))

                results["Long precision avg"] = ('%.4f' % (P_long_2.mean()))
                results["Long recall avg"] = ('%.4f' % (R_long_2.mean()))
                results["Long F1-score avg"] = ('%.4f' % (F1_long_2.mean()))
                results["Keyword precision avg"] = ('%.4f' % (P_key_2.mean()))
                results["Keyword recall avg"] = ('%.4f' % (R_key_2.mean()))
                results["Keyword F1-score avg"] = ('%.4f' % (F1_key_2.mean()))


            # Condition to look at one score
        else:
            long_summ = [long_summs[index]]
                ref_summ = [ref_summs[index]]
                short_summ = [short_summs[index]]
                key_ref_summ = [key_ref_summs[index]]

                P_long_1, R_long_1, F1_long_1 = scorer_1.score(long_summ, ref_summ, verbose=True)
                P_key_1, R_key_1, F1_key_1 = scorer_1.score(short_summ, key_ref_summ, verbose=True)

                P_long_2, R_long_2, F1_long_2 = scorer_2.score(long_summ, ref_summ, verbose=True)
                P_key_2, R_key_2, F1_key_2 = scorer_2.score(short_summ, key_ref_summ, verbose=True)

                results = {}

                results["Rescaled Long precision avg "] = ('%.4f' % (P_long_1))
                results["Rescaled Long recall avg"] = ('%.4f' % (R_long_1))
                results["Rescaled Long F1-score avg"] = ('%.4f' % (F1_long_1))
                results["Rescaled Keyword precision avg"] = ('%.4f' % (P_key_1))
                results["Rescaled Keyword recall avg"] = ('%.4f' % (R_key_1))
                results["Rescaled Keyword F1-score avg"] = ('%.4f' % (F1_key_1))

                results["Long precision avg"] = ('%.4f' % (P_long_2))
                results["Long recall avg"] = ('%.4f' % (R_long_2))
                results["Long F1-score avg"] = ('%.4f' % (F1_long_2))
                results["Keyword precision avg"] = ('%.4f' % (P_key_2))
                results["Keyword recall avg"] = ('%.4f' % (R_key_2))
                results["Keyword F1-score avg"] = ('%.4f' % (F1_key_2))

        # Condition if we only want to evaluate the long/non-tokenized versions of summaries
    else:
        # Condition to look at average long scores for whole dataset
            if index == None:
                P_long_1, R_long_1, F1_long_1 = scorer_1.score(long_summs, ref_summs, verbose=True)
                P_long_2, R_long_2, F1_long_2 = scorer_2.score(long_summs, ref_summs, verbose=True)

                results = {}
                results["Rescaled Long precision avg"] = ('%.4f' % (P_long_1.mean()))
                results["Rescaled Long recall avg"] = ('%.4f' % (R_long_1.mean()))
                results["Rescaled Long F1-score avg"] = ('%.4f' % (F1_long_1.mean()))

                results["Long precision avg"] = ('%.4f' % (P_long_2.mean()))
                results["Long recall avg"] = ('%.4f' % (R_long_2.mean()))
                results["Long F1-score avg"] = ('%.4f' % (F1_long_2.mean()))

            # Condition to look at one long score
        else:
            long_summ = [long_summs[index]]
                ref_summ = [ref_summs[index]]

                P_long_1, R_long_1, F1_long_1 = scorer_1.score(long_summ, ref_summ, verbose=True)
                P_long_2, R_long_2, F1_long_2 = scorer_2.score(long_summ, ref_summ, verbose=True)

                results = {}
                results["Rescaled Long precision avg"] = ('%.4f' % (P_long_1))
                results["Rescaled Long recall avg"] = ('%.4f' % (R_long_1))
                results["Rescaled Long F1-score avg"] = ('%.4f' % (F1_long_1))

                results["Long precision avg"] = ('%.4f' % (P_long_2))
                results["Long recall avg"] = ('%.4f' % (R_long_2))
                results["Long F1-score avg"] = ('%.4f' % (F1_long_2))

        return results


    def get_matrix(self, gen_summ, ref_summ, index, lang='fr'):

        '''
        Function to look at the relation matrix between any two generated and reference summaries
        '''

        if lang =='fr':
            scorer = self.scorer
        elif lang == 'en':
            scorer = self.scorer_en

        return scorer.plot_example(gen_summ[index], ref_summ[index])
