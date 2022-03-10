import os
import shutil
import subprocess

from tqdm import tqdm

subprocess.run(['python3', '-m', 'pip', 'install', 'tqdm'])

def get_articles(in_dataset, test=True): # Load reference dataset
    testset = None
    if in_dataset == 'MLSUM FR':
        from datasets import load_dataset
        dataset = load_dataset('mlsum', 'fr')
        testset = dataset['test']
        trainset = dataset['train']
    elif in_dataset == 'MLSUM EN':
        from datasets import load_dataset
        dataset = load_dataset('cnn_dailymail', '3.0.0')
        testset = dataset['test']
        trainset = dataset['train']
    elif in_dataset == 'ER/Actu':
        from newsjam.summ.text_processing import Pre
        pre = Pre()
        import json
        trainset = []
        with open('data/actu_articles.json', encoding='utf-8') as f:
            testset = json.load(f)
        import csv
        with open('data/lest_rep_updated_tagged.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                testset.append(row)
        print('Pre-processing corpus...')
        for article in tqdm(testset):
            article['text'] = pre.fr_phrases(article['text'])
    elif in_dataset == 'Guardian':
        from newsjam.summ.text_processing import Pre
        pre = Pre()
        import csv
        trainset = []
        testset = []
        with open('data/Guardian Data Final.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)    
            for row in reader:
                if 'text' in row.keys():
                    testset.append(row)
        print('Pre-processing corpus...')
        for article in tqdm(testset):
            article['text'] = pre.en_phrases(article['text'])
    
    # Prepare keyword sentences
    import spacy
    if in_dataset in ['MLSUM_FR', 'ER/Actu']:
        lang = 'fr'
        nlp = spacy.load("fr_core_news_lg")
    else:
        lang = 'en'
        nlp = spacy.load('en_core_web_trf')
    articles = testset if test else trainset
    return articles, nlp, lang

print('========================================')
print('Newsjam Experiment Running Driver (NERD)')
print('========================================')
print()

# Initial downloads
print('Retrieving Git repository...')
try:
    shutil.rmtree('newsjam')
except FileNotFoundError:
    print('No previous newsjam repo was found. Cloning.')
subprocess.run(['git', 'clone', 'https://github.com/pie3636/newsjam.git'])
os.chdir('newsjam')
print('Installing Python modules...')
subprocess.run(['python3', '-m', 'pip', 'install', '-r', 'requirements.txt'])
print('Downloading SpaCy models...')
subprocess.run(['python3', '-m', 'spacy', 'download', 'fr_core_news_lg'])
subprocess.run(['python3', '-m', 'spacy', 'download', 'en_core_web_trf'])
print()

# Read parameters
exp_type = ''
while exp_type not in ['e', 's']:
    print('- Summarization experiments generate summaries using one of')
    print('the implemented methods. They are typically very long (hours).')
    print('- Evaluation experiments compute the value of a particular metric')
    print('on generated summaries. Requires a previous summarization experiment.')
    print('Those are usually shorter (minutes to an hour).')
    print()
    exp_type = input('Do you want a [S]ummarization or [E]valuation experiment? ').lower()
    print()

filename_to_params = {
    'da': 'MLSUM FR',
    'db': 'MLSUM EN',
    'dc': 'ER/Actu',
    'dd': 'Guardian',
    'ma': 'LSA/LSI',
    'mb': 'k-means (FlauBERT)',
    'mc': 'k-means (CamemBERT)',
    'md': 'k-means (RoBERTa)',
    '0': 'No',
    '1': 'Yes'
}

params_to_filename = {v: k for k, v in filename_to_params.items()}

metrics = ['ROUGE-L', 'BERTScore', 'Word Mover distance', 'Time measurement']
methods = ['LSA/LSI', 'k-means (FlauBERT)', 'k-means (CamemBERT)', 'k-means (RoBERTa)']
datasets = ['MLSUM FR', 'MLSUM EN', 'ER/Actu', 'Guardian']

timing = False

# Input dataset handling
if exp_type == 'e':
    print('The following files are available (in the GitHub repository, "gen" folder):')
    print()
    in_files = list(os.listdir('gen'))
    print('ID  Filename            Dataset   Method                   Pretraining    Keywords')
    print('----------------------------------------------------------------------------------')
    for i, in_file in enumerate(in_files):
        d, m, p, k = [filename_to_params[x] for x in in_file.split('.')[0].split('_')]
        print('{}'.format(i).ljust(3) + ' {}'.format(in_file).ljust(20) + ' {}'.format(d).ljust(10) + ' {}'.format(m).ljust(25) + ' {}'.format(p).ljust(15) + ' {}'.format(k))
    print()
    file_name = ''
    while file_name not in range(len(in_files)):
        file_name = input('Filename ID selection? ')
        try:
            file_name = int(file_name)
        except ValueError:
            pass
    in_file = in_files[file_name]
    in_dataset, sum_method, pretraining, keywords = [filename_to_params[x] for x in in_file.split('.')[0].split('_')]
    print()
    print('The following evaluation methods are available:')
    print('0 - ROUGE-L')
    print('1 - BERTScore')
    print('2 - Word Mover distance')
    print('3 - Time measurement')
    print()
    metric = ''
    while metric not in ['0', '1', '2', '3']:
        metric = input('Chosen metric: ')
    
    metric = metrics[int(metric)]
    
    if metric == 'Word Mover distance':
        pretraining2 = ''
        while pretraining2 not in ['y', 'n']:
            pretraining2 = input('Should embeddings be fine-tuned for Word Mover Distance evaluation? [Y/N] ').lower()
        pretraining2 = pretraining2 == 'y'
    print()
    
    # Read generated summaries
    with open('gen/' + in_file, encoding='utf-8') as f:
        data = f.read().split('\n')
        gen_summs = []
        cur_summ = []
        count = 0
        for line in data:
            if line.strip() == str(count):
                gen_summs.append(cur_summ[:-1])
                cur_summ = ''
                count += 1
            else:
                cur_summ += line + '\n'
        gen_summs.append(cur_summ[:-1])
        gen_summs = gen_summs[1:]
    
    # Read second list of generated summaries
    other = in_file.replace('_0.txt', '_2.txt').replace('_1.txt', '_0.txt').replace('_2.txt', '_1.txt')
    with open('gen/' + other, encoding='utf-8') as f:
        data = f.read().split('\n')
        other_gen_summs = []
        cur_summ = []
        count = 0
        for line in data:
            if line.strip() == str(count):
                other_gen_summs.append(cur_summ[:-1])
                cur_summ = ''
                count += 1
            else:
                cur_summ += line + '\n'
        other_gen_summs.append(cur_summ[:-1])
        other_gen_summs = other_gen_summs[1:]
    
    articles, nlp, lang = get_articles(in_dataset) # Read reference dataset

    # Prepare keyword sentences
    if 'summary' in article[0]:
        ref_summs = [article['summary'] for article in articles]
    else:
        ref_summs = [article['highlights'] for article in articles]
    other_ref_summs = ['\n'.join(get_keyword_sentences(nlp(summary), lang)) for summary in ref_summs]
    
    if keywords:
        gen_summs, other_gen_summs = other_gen_summs, gen_summs
        
    # Compute metric
    if metric == 'ROUGE-L':
        from newsjam.eval.rouge_l import RougeLEval
        rouge_l_eval = RougeLEval()
        scores1, scores2 = rouge_l_eval.evaluate_many(zip(ref_summs, other_ref_summs), zip(gen_summs, other_gen_summs))
        results = rouge_l_eval.get_results(scores1, scores2)
    elif metric == 'BERTScore':
        from newsjam.eval.bert_eval import BERT_Eval
        bert_eval = BERT_Eval()
        results = bert_eval.bert_score(gen_summs, ref_summs, other_gen_summs, other_ref_summs, lang=lang)
    elif metric == 'Word Mover distance':
        from newsjam.eval.wordmover import WordMoverEval
        import fasttext.util
        model = fasttext.load_model(f'cc.{lang}.300.bin')
        if pretraining2:
            if 'text' in article:
                orig_sents = [nlp(article['text']).sents for article in articles]
            else:
                orig_sents = [nlp(article['article']).sents for article in articles]
            pretraining = {'epochs': 5, 'sents': orig_sents}
        else:
            pretraining = {}
        wordmover_eval = WordMoverEval(model, finetuning2, fine_tune=pretraining)
        results = wordmover_eval.evaluate_many(ref_summs, gen_summs)
    elif metric == 'Time measurement':
        timing = True
    
    # Display results
    if not timing:
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print('Please post the following block of text to the #eval-results channel on Discord:')
        print()
        print('='*50)
        print(f'Results for {in_file} / {metric}' + (f' (pretrained)' if pretraining2 else ''))
        print('='*50)
        for k, v in results.items():
            print(k.ljust(25), round(v*100, 3), '%')
    

if exp_type == 's' or timing:
    # Read parameters
    print('The following summarization methods are available:')
    print('0 - LSA/LSI')
    print('1 - Embeddings + k-means clustering (FlauBERT)')
    print('2 - Embeddings + k-means clustering (CamemBERT)')
    print('3 - Embeddings + k-means clustering (RoBERTa)')
    print()
    method = ''
    while method not in ['0', '1', '2', '3']:
        method = input('Chosen method: ')
    
    method = methods[int(method)]
    
    print()
    print('The following datasets are available:')
    print('0 - MLSUM FR')
    print('1 - MLSUM EN')
    print('2 - ER/Actu')
    print('3 - Guardian')
    in_dataset = ''
    while in_dataset not in ['0', '1', '2', '3']:
        in_dataset = input('Chosen dataset: ')
    
    in_dataset = datasets[int(in_dataset)]
    
    pretraining = False
    if method != 'LSA/LSI':
        while pretraining not in ['y', 'n']:
            pretraining = input('Should input embeddings be pre-trained? [Y/N] ').lower()
        pretraining = pretraining == 'y'
    print()
    
    # Read dataset and load summarizer
    articles, nlp, lang = get_articles(in_dataset)
    
    if method == 'LSA/LSI':
        from newsjam.summ.lsa import LSASummarizer
        summ = LSASummarizer()
    elif method == 'k-means (FlauBERT)':
        from newsjam.summ.bert_embed import BertEmbeddingsSummarizer
        summ = BertEmbeddingsSummarizer('flaubert/flaubert_large_cased')
    elif method == 'k-means (CamemBERT)':
        from newsjam.summ.bert_embed import BertEmbeddingsSummarizer
        summ = BertEmbeddingsSummarizer('camembert/camembert-large')
    elif method == 'k-means (RoBERTa)':
        from newsjam.summ.bert_embed import BertEmbeddingsSummarizer
        summ = BertEmbeddingsSummarizer('roberta-base')
    
    # Pre-train BERT model
    if pretraining:
        trainset, _, _ = get_articles(in_dataset, False) # TODO
    
    # Run summarization
    gen_summs = []
    
    if timing:
        from timeit import default_timer as timer
        start = timer()
    
    for article in tqdm(articles):
        if 'text' in article:
            gen_summs.append(summ.get_summary(article['text']))
        else:
            gen_summs.append(summ.get_summary(article['article']))
    
    # Display time measurement results
    if timing:
        end = timer()
        duration = end - start
        results = {
            'Time': duration,
            'Time/article': duration/len(articles),
            'Time/1000 chars': 1000*duration/sum(len([x['text'] if 'text' in article else x['article'] for x in articles]))
        
        }
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print('Please post the following block of text to the #eval-results channel on Discord:')
        print()
        print('='*50)
        print(f'Results / Time measurement')
        print('='*50)
        for k, v in results.items():
            print(k.ljust(25), round(v, 3), 's')
    
        print()
        print('Press [Return] once this is done.')
        input()
    
    # Save data to file
    filename = os.path.expanduser('~/') + '_'.join([params_to_filename[x] for x in [in_dataset, method, "Yes" if pretraining else "No"]])
    filename_full = filename + '_0.txt'
    filename_kw = filename + '_1.txt'
    f = open(filename_full, 'w')
    f2 = open(filename_kw, 'w')
    for i, (full_summ, kw_summ) in enumerate(gen_summs):
        f.write(str(i) + '\n')
        f.write(full_summ + '\n')
        f2.write(str(i) + '\n')
        f2.write(kw_summ + '\n')
    f.close()
    f2.close()
    
    import getpass
    username = getpass.getuser()
    
    print('='*80)
    print()
    print('===================')
    print('EXPERIMENT COMPLETE')
    print('===================')
    print()
    print('The experiment output has been saved to the following two files:')
    print()
    print(filename_full)
    print(filename_kw)
    print()
    print('IMPORTANT: Please retrieve them using the following command')
    print('(on your local computer, once disconnected from grid5k):')
    print()
    print(f'scp "{username}@access.grid5000.fr:/home/{username}/{filename_full}" ~; scp "{username}@access.grid5000.fr:/home/{username}/{filename_kw}" ~')
    print()
    print('The files will be saved to the user/home folder of your local computer.')
    print('Then upload them to the gen/ folder on GitHub.')
    print('='*80)
    