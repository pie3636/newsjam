from timeit import default_timer as timer
from .eval import Eval

from tqdm import tqdm

class TimeEval(Eval):
    def __init__(self):
        super().__init__()
    
    
    def evaluate_one(self, text, method, lang, *args, **kwargs):
        """
            Measures the summarization time of a method on one article
            Arguments:
                `text`      The text to summarize
                `method`    The summarization class to use
                `lang`      Language code of the text to summarize
                `args`      Positional arguments to pass to `method`
                `kwargs`    Keyword arguments to pass to `method`
            Returns the execution time of the summarization method
        """
        
        start = timer()
        method_instance = method(*args, **kwargs)
        method_instance.get_summary(text, lang=lang)
        end = timer()
        return end - start


    def evaluate_many(self, texts, lang, method, *args, **kwargs):
        """
            Measures the average summarization time for all articles in a set
            Arguments:
                `texts`     A list of all texts to summarize
                `method`    The summarization class to use
                `lang`      Language code of the text to summarize
                `args`      Positional arguments to pass to `method`
                `kwargs`    Keyword arguments to pass to `method`
            Returns the average execution time of the summarization method
        """
        start = timer()
        method_instance = method(*args, **kwargs)
        for text in tqdm(texts):
            method_instance.get_summary(text, lang=lang)
        end = timer()
        return (end - start)/len(texts)
