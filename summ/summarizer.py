# SpaCy model for segmentation and tokenization
import spacy

class Summarizer():
    def __init__(self, max_len=280):
        self.nlp = spacy.load("fr_core_news_lg")
        self.nlp_en = spacy.load('en_core_web_trf')
        self.max_len = max_len
    
    def get_summary(self, article):
        raise NotImplementedError('<Summarizer> is an abstract class. Please instantiate a child class to call this method.')
    