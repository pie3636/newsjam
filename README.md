# newsjam
Jammin' the news™ <sub><sup>tabarnak</sup></sub>

## Repository structure:
- `main.ipynb` contains the main module. It can instantiate specific summarization and evaluation submodules, as well as save generated summaries to an output file.
- `data\` contains all the scraping notebooks as well as their output in JSON format:
  - `\est_republicain.ipynb` contains the scraping part for L'Est Républicain/
  - `\est_republicain.json` contains the JSON-formatted list of articles extracted from L'Est Républicain.
  - `\actu_preliminary.json` contains the JSON-formatted list of articles extracted from Actu.
- `eval\` contains all the modules implementing evaluation metrics:
  - `\rouge_l.py` contains the implementation of the ROUGE-L score.
- `summ\` contains all the modules implementing summarization methods:
  - `\lsa.py` contains the implementation of summarization using Latent Semantic Analysis
  - `\lsa.ipynb` contains a notebook version of the previous implementation, made to be more readable and interactive, so that everyone can run the programs step by step and see the output of each section.

## Evaluation

### Basic LSA (sprint 1):
Evaluation was made using ROUGE-L scores between the reference summary and the generated summary.
Scores have been computed in two ways:
- The `long` scores were measured for the full raw text of both summaries. The punctuation, spacing and words are all unedited.
- The `keyword` scores were measured on both summaries after stripping them of punctuation (except sentence-delimiting periods), with stopword removal and word lemmatization.

#### Results
- Actu corpus (scraped, 47 articles):
```
Long precision avg        50.826 %
Keyword precision avg     49.695 %
Long recall avg           58.922 %
Keyword recall avg        58.561 %
Long F1-score avg         53.913 %
Keyword F1-score avg      53.086 %
```

- MLSUM corpus, test set (15828 articles):
```
Long precision avg        14.641 %
Keyword precision avg     11.038 %
Long recall avg           16.659 %
Keyword recall avg        12.896 %
Long F1-score avg         15.071 %
Keyword F1-score avg      11.465 %
```
