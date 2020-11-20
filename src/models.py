import os
import json
from tqdm import tqdm
from os.path import join as pjoin
from multiprocessing import pool
from lexrank import LexRank, STOPWORDS
from nltk import sent_tokenize

from utils import test_rouge, read_line, rouge_results_to_str, read_json

class Lead(object):
    def __init__(self, data_path, data_type, extract_num=3, processors=8):
        self.extract_num = extract_num
        self.processors = processors
        self.full_path = pjoin(data_path, f'{data_type}.json')
    
    def extract_summary(self, ):
        data = read_json(self.full_path)

        articles = []
        abstracts = []

        for item in data:
            articles.append(item['article'])
            abstracts.append(item['abstract'])
        data_iterator = zip(articles, abstracts)

        summaries = []
        references = []

        for item in tqdm(data_iterator, desc="Lead:"):
            article, abstract = item
            summary = article[:self.extract_num]
            summaries.append(summary)
            references.append([abstract])

        result = test_rouge(summaries, references, self.processors)
        return result

class TextRank(object):
    def __init__(self, data_path, data_type, extract_ratio=0.2, processors=8):
        self.extract_ratio = extract_ratio
        self.processors = processors

        self.full_path = pjoin(data_path, f'{data_type}.json')
        

    def extract_summary(self, ):
        data = read_json(self.full_path)
        articles = []
        abstracts = []

        for item in data:
            articles.append("\n".join(item['article']))
            abstracts.append([item['abstract']])
        summaries = []

        for article in tqdm(articles, desc="Text Rank:"):
            summaries.append(summarize(article, split=True, ratio=self.extract_ratio))

        res = test_rouge(summaries, abstracts, self.processors)
        return res

class LexRank_text(object):
    def __init__(self, data_path, data_type, summary_size=3, threshold=.1, processors=8):
        self.summary_size = summary_size
        self.threshold = threshold
        self.processors = processors

        self.full_path = pjoin(data_path, f'{data_type}.json')
        

    def extract_summary(self, ):
        data = read_json(self.full_path)
        articles = []
        abstracts = []

        for item in data:
            articles.append(item['article'])
            abstracts.append([item['abstract']])
        
        lxr = LexRank(articles, stopwords=STOPWORDS['en'])
 
        summaries = [lxr.get_summary(x, summary_size=self.summary_size, threshold=self.threshold) for x in tqdm(articles, desc="LexRank:")]
        res = test_rouge(summaries, abstracts, self.processors)

        return res

class Oracle(object):
    def __init__(self, data_path, data_type, processors=8):
        self.processors = processors
        self.full_path = pjoin(data_path, f'{data_type}.json')
    
    def extract_summary(self, ):
        data = read_json(self.full_path)

        articles = []
        abstracts = []
        oracle_ids = []

        for item in data:
            articles.append(item['article'])
            abstracts.append(item['abstract'])
            if 'NYT' in self.full_path:
                oracle_ids.append(item['oracle_sens'])
            else:
                oracle_ids.append(item['oracle'])
        data_iterator = zip(articles, abstracts, oracle_ids)
        # print(data_iterator)

        summaries = []
        references = []

        for item in tqdm(data_iterator, desc="Oracle:"):
            article, abstract, oracle_id = item
            summary = [article[x] for x in oracle_id]
            summaries.append(summary)
            references.append([abstract])


        # result = evaluate_rouge(summaries, references, remove_temp=True, rouge_args=[])
        result = test_rouge(summaries, references, self.processors)
        return result



if __name__ == "__main__":

    data_path = 'data/cnndm'
    # data_path = '../mypacsum/data/NYT'
    data_type = 'test'

    processors = 8

    # model = Oracle(data_path, data_type, processors=processors)
    # model = LexRank_text(data_path, data_type, processors=processors)
    model = Lead(data_path, data_type, processors=processors)
    model.extract_summary()

    




