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
#from eval.rouge_l import RougeLEval
#from summ.lsa import LSASummarizer

import spacy
from spacy.lang.fr.stop_words import STOP_WORDS

# BERTScore import
from bert_score import BERTScorer

from tqdm import tqdm

dataset = load_dataset('mlsum', 'fr')

#rouge_l = RougeLEval()
#lsa_summ = LSASummarizer()


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

def bert_score(ref_summ, gen_summ):
    nlp = spacy.load("fr_core_news_sm")

    # Loop to separate long and short summaries from dataset
    # tried just unpacking the two variables from gen_summ like in rouge_l.py but it wasn't working for me
    long_summs = []
    short_summs = []
    for x in range(len(gen_summ)):
        long_summs.append(gen_summ[x][0])
        short_summs.append(gen_summ[x][1])

    summ_sentences = []
    summ_cur_sentence = []
    for summ in ref_summ:
        summ = nlp(summ)
        for sent in summ.sents:
            for token in sent:
                if not token.text.lower() in STOP_WORDS and not token.is_punct:
                    summ_cur_sentence.append(token.lemma_)
            summ_sentences.append(summ_cur_sentence)
            summ_cur_sentence = []

    key_ref_summs = [' '.join(sent) for sent in summ_sentences]

    scorer = BERTScorer(lang='fr', rescale_with_baseline=True)

    P_long, R_long, F1_long = scorer.score(long_summs, ref_summ, verbose=True)
    #P_key, R_key, F1_key = scorer.score(short_summs, key_ref_summs, verbose=True)
    # P = precision
    # R = recall
    # F1 = F1-score

    results = {}
    results["Long precision avg"] = P_long.mean()
    results["Long recall avg"] = R_long.mean()
    results["Long F1-score avg"] = F1_long.mean()
    #results["Keyword precision avg"] = P_key.mean()
    #results["Keyword recall avg"] = R_key.mean()
    #results["Keyword F1-score avg"] = F1_key.mean()

    return results

bert_score(ref_summs, gen_summs)

# Will try to get working with function above, so we can see keyword matrix as well

#scorer.plot_example(gen_summs[0], ref_summs[0])