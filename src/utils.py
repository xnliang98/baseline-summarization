import os
import sys
import codecs
import json
import random
import string
import shutil
import time

import pickle as pkl

from multiprocessing import Pool

from pyrouge import Rouge155

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} not exists, we have created it.")
    return True

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    return lines

def read_line(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def write_line(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.writelines(item + '\n')
    print(f"Data has saved to {path}.")
    return True

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    print(f"Data has saved to {path}.")
    return True

def evaluate_rouge(data, remove_temp=True, rouge_dir='/root/env/pyrouge/tools/ROUGE-1.5.5/', rouge_args='-e /root/env/pyrouge/tools/ROUGE-1.5.5/data -c 95 -r 1000 -n 2 -a'):
    summaries, references = data

    temp_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    temp_dir = os.path.join("temp",temp_dir)
    print(temp_dir)
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    print(temp_dir, system_dir, model_dir)

    assert len(summaries) == len(references)
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        summary_fn = '%i.txt' % i
        for j, candidate in enumerate(candidates):
            candidate_fn = '%i.%i.txt' % (i, j)
            with open(os.path.join(model_dir, candidate_fn), 'w') as f:
                f.write('\n'.join(candidate))
    
        with open(os.path.join(system_dir, summary_fn), 'w') as f:
            f.write('\n'.join(summary))

    args_str = ' '.join(map(str, rouge_args))
    rouge = Rouge155(rouge_dir=rouge_dir)
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    rouge.system_filename_pattern = '(\d+).txt'
    rouge.model_filename_pattern = '#ID#.\d+.txt'
    
    output = rouge.convert_and_evaluate(rouge_args=rouge_args)

    r = rouge.output_to_dict(output)
    print(output)

    # remove the created temporary files
    if remove_temp:
       shutil.rmtree(temp_dir)
    return r

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def test_rouge(candidates, references, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    # candidates = [line.strip() for line in cand]
    # references = [line.strip() for line in ref]

    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)

    candidates_chunks = list(chunks(candidates, int(len(candidates)/num_processes)))
    references_chunks = list(chunks(references, int(len(references)/num_processes)))
    n_pool = len(candidates_chunks)

    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i], references_chunks[i]))
    pool = Pool(n_pool)

    results = pool.map(evaluate_rouge, arg_lst)

    final_results = {}
    for i, r in enumerate(results):
        for k in r:
            if(k not in final_results):
                final_results[k] = r[k] * len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k] / len(candidates)
    
    print(rouge_results_to_str(final_results))
    return final_results
    
def rouge_results_to_str(results_dict):
    return "ROUGE-F(1/2/3/l): {:.2f}  {:.2f}  {:.2f}\nROUGE-R(1/2/3/l): {:.2f}  {:.2f}  {:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
    results_dict["rouge_1_recall"] * 100,
    results_dict["rouge_2_recall"] * 100,
    # results_dict["rouge_3_f_score"] * 100,
    results_dict["rouge_l_recall"] * 100
    # ,results_dict["rouge_su*_f_score"] * 100
    )


# if __name__ == "__main__":
#     # init_logger('test_rouge.log')
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', type=str, default="candidate.txt",
#                         help='candidate file')
#     parser.add_argument('-r', type=str, default="reference.txt",
#                         help='reference file')
#     parser.add_argument('-p', type=int, default=8,
#                         help='number of processes')
#     args = parser.parse_args()
#     print(args.c)
#     print(args.r)
#     print(args.p)
#     if args.c.upper() == "STDIN":
#         candidates = sys.stdin
#     else:
#         candidates = codecs.open(args.c, encoding="utf-8")
#     references = codecs.open(args.r, encoding="utf-8")

#     results_dict = test_rouge(candidates, references, args.p)
#     # return 0
#     print(time.strftime('%H:%M:%S', time.localtime())
# )
#     print(rouge_results_to_str(results_dict))
