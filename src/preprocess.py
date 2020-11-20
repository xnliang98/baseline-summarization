import os
import re
import json
from os.path import join as pjoin

from tqdm import tqdm
from multiprocessing import Pool
from nltk import sent_tokenize, word_tokenize
from utils import read_line, write_line, _get_ngrams, _get_word_ngrams, cal_rouge


def greedy_selection(data):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    doc_sent_list, abstract_sent_list, summary_size = data
    max_rouge = 0.0

    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge
    return sorted(selected)

def split_sentence(sentence):
    return sent_tokenize(sentence.strip())

def tokenize(sentences):
    return [word_tokenize(x) for x in sentences]

def convert_line_to_json(data_path, data_type, processers=1):
    source_path = pjoin(data_path, f'{data_type}.source')
    target_path = pjoin(data_path, f'{data_type}.target')

    test_source = read_line(source_path)
    test_target = read_line(target_path)

    assert len(test_source) == len(test_target)

    pool = Pool(processers)
    articles = pool.map(split_sentence, test_source)
    abstracts = pool.map(split_sentence, test_target)
    assert len(articles) == len(abstracts)

    new_articles = pool.map(tokenize, articles)
    new_abstracts = pool.map(tokenize, abstracts)

    data = zip(new_articles, new_abstracts, [10] * len(new_abstracts))

    oracle_ids = pool.map(greedy_selection, data)
    
    import collections
    lens = [len(oracle_id) for oracle_id in oracle_ids]
    print(collections.Counter(lens))
    
    doc_id = 0
    json_datas = []
    for article, abstract, oracle_id in zip(articles, abstracts, oracle_ids):
        json_data = {
            'doc_id': doc_id,
            'article': article,
            'abstract': abstract,
            'oracle': oracle_id
        }
        json_datas.append(json.dumps(json_data))
    
    write_line(json_datas, pjoin(data_path, f"{data_type}.json"))

if __name__ == "__main__":
    import sys
    data_type = sys.argv[2]
    data_path = sys.argv[1]
    processers = int(sys.argv[3])
    convert_line_to_json(data_path, data_type, processers)




