# Not a final version of the BERTScore implemenation; this was just to test it out on the scraped data

# For the first execution, you will need to uncomment this line
# to download the SpaCy model and other necessary packages. Then you can comment it back
# !python -m spacy download fr_core_news_sm
# !python -m pip install ipynb

# MLSUM Corpus
from datasets import load_dataset

# Loading article data
import json

# Our packages
from eval.rouge_l import RougeLEval
from summ.lsa import LSASummarizer
from bert_score import BERTScorer

from tqdm import tqdm

dataset = load_dataset('mlsum', 'fr')

rouge_l = RougeLEval()
lsa_summ = LSASummarizer()


# Summarization run on scraped data
with open('data/actu_preliminary.json', 'r', encoding='utf-8') as jsonfile:
    data = json.load(jsonfile)

texts = [article['text'] for article in data]
ref_summs = [article['summary'] for article in data]

gen_summs = []
for text in tqdm(texts):
    gen_summs.append(lsa_summ.get_summary(text))

scores1, scores2 = rouge_l.evaluate_many(ref_summs, gen_summs)
results = rouge_l.get_results(scores1, scores2)

for k, v in results.items():
    print(k.ljust(25), round(v*100, 3), '%')


# BERTScore Implementation
# you can copy this code to the bottom of main.ipynb to see the outputs

# loop to get a list of only the unstemmed extracted summaries
cand_summs = []
for x in range(len(gen_summs)):
    cand_summs.append(gen_summs[x][0])

scorer = BERTScorer(lang='fr', rescale_with_baseline=False)

P, R, F1 = scorer.score(cand_summs, ref_summs, verbose=True)
print(P, R, F1, sep='\n')
# P = precision
# R = recall
# F1 = F1

# Plots a similarity matrix showing the relation between all words in extracted summary to all words in reference summary
scorer.plot_example(cand_summs[3], ref_summs[3])