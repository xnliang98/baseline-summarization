# Baseline Models for Unsupervised Summarization
Data: xxx.source, xxx.target (one document one line).

## Requirements
- pyrouge (need setup: https://github.com/xnliang98/pyrouge-setup)
- gensim
- nltk
- tqdm
- lexrank

## preprocess data from xxx.source, xxx.target to json file
Need to modeify params: `rouge_dir` and `rouge_args` in `utils.evaluate_rouge` method.
```shell
python src/preprocess.py data_path xxx num_cpus
```

## Models
- Lead-3
- Oracle
- TextRank
- LexRank
- [Waiting reconstruct] MMR, From https://github.com/Alex-Fabbri/Multi-News/tree/master/code/extractive_code/mmr. 

How to run models:
```python
model = ClassName(data_path, data_type, processors, **)
model.extract_summary()
```

How to run MMR:
```shell
python src/mmr.py data_path xxx
```


