import os
import subprocess

print('========================================')
print('Newsjam Experiment Running Driver (NERD)')
print('========================================')
print()

print('Retrieving Git repository...')
subprocess.call('git', 'clone', 'https://github.com/pie3636/newsjam.git')
os.chdir('newsjam')
print('Installing Python modules...')
subprocess.call('python3', '-m', 'pip', 'install', '-r', 'requirements.txt');
print('Downloading SpaCy models...')
subprocess.call('python3', '-m', 'spacy', 'download', 'fr_core_news_lg');
subprocess.call('python3', '-m', 'spacy', 'download', 'en_core_web_trf');
print()

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

filename_converter = {
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

metrics = {
    '0': 'ROUGE-L',
    '1': 'BERTScore',
    '2': 'Word Mover distance',
    '3': 'Time measurement'
}

if exp_type == 'e':
    print('The following datasets are available (in the GitHub repository, "gen" folder):')
    print()
    in_files = list(os.listdir('gen'))
    print('ID  Filename            Dataset   Method                   Finetuning     Keywords')
    print('----------------------------------------------------------------------------------')
    for i, in_file in enumerate(in_files):
        d, m, f, k = [filename_converter[x] for x in in_file.split('.')[0].split('_')]
        print('{}'.format(i).ljust(3) + ' {}'.format(in_file).ljust(20) + ' {}'.format(d).ljust(10) + ' {}'.format(m).ljust(25) + ' {}'.format(f).ljust(15) + ' {}'.format(k))
    print()
    file_name = ''
    while file_name not in range(len(in_files)):
        file_name = input('Filename ID selection? ')
        try:
            file_name = int(file_name)
        except ValueError:
            pass
    in_file = in_files[file_name]
    in_dataset, sum_method, finetuning, keywords = [filename_converter[x] for x in in_file.split('.')[0].split('_')]
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
    
    metric = metrics[metric]
    
    if metric == 'Word Mover distance':
        finetuning2 = ''
        while finetuning2 not in ['y', 'n']:
            finetuning2 = input('Should embeddings be fine-tuned for Word Mover Distance evaluation? [Y/N]').lower()
        finetuning2 = finetuning2 == 'y'
        
    
    
    print()
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
    
    if in_dataset == 'MLSUM FR':
        from datasets import load_dataset
        articles = load_dataset('mlsum', 'fr')['test']
    elif in_dataset == 'MLSUM EN':
        from datasets import load_dataset
        articles = load_dataset('cnn_dailymail', '3.0.0')['test']
    elif in_dataset == 'ER/Actu':
        import json
        with open('data/actu_articles.json', encoding='utf-8') as f:
            articles = json.load(f)
        import csv
        with open('data/lest_rep_updated_tagged.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            next(reader)
            for row in reader:
                articles.append(row)
    elif in_dataset == 'Guardian':
    import csv
        articles = []
        with open('data/Guardian Data Final.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            next(reader)
            for row in reader:
                articles.append(row)
    print(len(articles))
    # sum_method, finetuning, keywords, finetuning2

else:
    pass # TODO Summarize