import spacy

class Eval():
    """
        Base class for all evaluation metrics
    """
    
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_lg")
        self.nlp_en = spacy.load('en_core_web_trf')
    
    
    def evaluate_one(self, ref_summ, gen_summ):
        raise NotImplementedError('<Eval> is an abstract class. Please instantiate a child class to call this method.')


    def evaluate_many(self, ref_summs, gen_summs, num_articles=None):
        raise NotImplementedError('<Eval> is an abstract class. Please instantiate a child class to call this method.')


    def get_results(self, long_scores, keyword_scores):
        raise NotImplementedError('<Eval> is an abstract class. Please instantiate a child class to call this method.')
