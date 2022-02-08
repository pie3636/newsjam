# *Newsjam*
Jammin' the news™

This repository contains the code, data, results and deliverables for the *Newsjam* group of the 703 Project Management course of the NLP master's degree at the IDMC, Université de Lorraine.

## Repository structure:
- `Annotation_Stats.ipynb` contains the IAA computation module.
- `Pipeline.ipynb` contains the full bot pipeline implementation, from scraping to posting. It requires a Twitter API key to run, which is not included in this repository for security reasons.
- `Text_Post-processing.py` contains the text post-processing function.
- `main.ipynb` contains the main summarization module. It can instantiate specific summarization and evaluation submodules, as well as save generated summaries to an output file.
- `classif\` contains data, scripts and notebooks related to the classification subtask:
  - `\Annotation Guidelines.docx` contains the annotation guidelines
  - `\csv_lest_republicain_summ.csv` contains the *L'Est Républicain* corpus and its manual tags
  - `\log_reg_classifier.ipynb`, `\doc_classification_logistic.ipynb` and `\naive_bayes.ipynb` implement various classification methods
  - `\log_reg_classifier.py` is the final classifier used in the pipeline
- `data\` contains data, scripts and notebooks related to scraping. In particular:
  - `\est_republicain.ipynb` contains the scraping functions for *L'Est Républicain*
  - `\est_republicain.json` contains the JSON-formatted list of articles extracted from *L'Est Républicain*
  - `\scraper_functions.ipynb` contains the scraping functions for *Actu*
  - `\scraper_functions.py` contains the final scraper for *Actu* that is used in the pipeline
  - `\actu_articles.json` contains the JSON-formatted list of articles extracted from *Actu*
- `deliver\` contains all reports, slides and posters that were delivered during the project's lifetime
- `eval\` contains all the modules implementing evaluation metrics:
  - `\bert_eval.py` contains the implementation of BERTScore
  - `\eval.py` is a helper file containing generic evaluation functions
  - `\rouge_l.py` contains the implementation of the ROUGE-L score
  - `\time.py` contains the implementation of the running time measurement
- `gen\` contains the generated summaries by all three summarizers when ran on our own corpus, in `full` and keyword-only (`kw`) versions
- `summ\` contains all the modules implementing summarization methods:
  - `\bert_embed.py` contains the implementation of summarization using BERT-like models and K-means clustering
  - `\lsa.py` contains the implementation of summarization using Latent Semantic Analysis
  - `\lsa.ipynb` contains a notebook version of the previous implementation, made to be more readable and interactive, so that everyone can run the programs step by step and see the output of each section.
  - `\sum_transformers.ipynb` contains an alternate implementation of summarization using BERT-like models
  - `\utils.py` contains various utility functions for summarization
