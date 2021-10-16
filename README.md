# newsjam
Jammin' the news™ <sub><sup>tabarnak</sup></sub>

## Main files:
- `LSA-Run.ipynb` contains the summarization part, currently implemented using Latent Semantic Analysis. It will automatically summarize one or several articles and compute the ROUGE-L scores.
- `LSA.ipynb` is similar but is made to be more readable and interactive so that everyone can run the program step by step and see the output of each section.
- `scraping.ipynb` contains the scraping part for L'Est Républicain

- `actu-preliminary.json` contains the JSON-formatted list of articles extracted from Actu.
- `lest_republicain.json` contains the JSON-formatted list of articles extracted from L'Est Républicain.

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
