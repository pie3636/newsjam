from timeit import default_timer as timer
import eval

class TimeEval(Eval):
	def __init__(self):
		super().__init__()
    
    
    def evaluate_one(self, text, method):
		"""
			Measures the summarization time of a method on one article
			Arguments:
				`text`	    The text to summarize
				`method`    The summarization class to use
			Returns the execution time of the summarization method
		"""
        
        start = timer()
        method_instance = method()
        method_instance.get_summary(text)
        end = timer()
        return end - start


	def evaluate_many(self, texts, method):
		"""
			Measures the average summarization time for all articles in a set
            Arguments:
				`texts`	    A list of all texts to summarize
				`method`    The summarization class to use
			Returns the average execution time of the summarization method
		"""
        start = timer()
		method_instance = method()
        for text in texts:
            method_instance.get_summary(text)
        end = timer()
        return (end - start)/len(texts)
