import re
import os
import copy
import sys
import nltk
import math
import string

from os.path import join as pjoin
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize

from utils import read_json, test_rouge

class Sentence(object):

	def __init__(self, preproWords, originalWords):
		self.preproWords = preproWords
		self.wordFrequencies = self.sentenceWordFreq()
		self.originalWords = originalWords
	
	def getPreProWords(self):
		return self.preproWords
	
	def getOriginalWords(self):
		return self.originalWords

	def getWordFreq(self):
		return self.wordFrequencies	
	
	def sentenceWordFreq(self):
		wordFreq = {}
		for word in self.preproWords:
			if word not in wordFreq.keys():
				wordFreq[word] = 1
			else:
				# if word in stopwords.words('english'):
				# 	wordFreq[word] = 1
				# else:			
				wordFreq[word] = wordFreq[word] + 1
		return wordFreq

  


counter_sim_devide0 = 0 
sentence_sim_exception = 0.000000000000000000001

def processFile(sample):
	lines = sample
	# setting the stemmer
	sentences = []
	porter = nltk.PorterStemmer()

	# modelling each sentence in file as sentence object
	for line in lines:
		# original words of the sentence before stemming
		originalWords = line[:]
		line = line.strip().lower()

		# word tokenization
		sent = nltk.word_tokenize(line)
		
		# stemming words
		stemmedSent = [porter.stem(word) for word in sent]		
		# stemmedSent = filter(lambda x: x!='.'and x!='`'and x!=','and x!='?'and x!="'" 
		# 	and x!='!' and x!='''"''' and x!="''" and x!="'s", stemmedSent)
		
		# list of sentence objects
		if stemmedSent != []:
			# sentences.append(sentence.sentence(file_name, stemmedSent, originalWords))	
			sentences.append(Sentence(stemmedSent, originalWords))			
	
	return sentences

def TFs(sentences):
	# initialize tfs dictonary
	tfs = {}

	# for every sentence in document cluster
	for sent in sentences:
		# retrieve word frequencies from sentence object
		wordFreqs = sent.getWordFreq()
	    
		# for every word
		for word in wordFreqs.keys():
			# if word already present in the dictonary
			if tfs.get(word, 0) != 0:				
				tfs[word] = tfs[word] + wordFreqs[word]
	        # else if word is being added for the first time
			else:				
				tfs[word] = wordFreqs[word]	
	return tfs

def IDFs(sentences):
    N = len(sentences)
    idf = 0
    idfs = {}
    words = {}
    w2 = []

    # every sentence in our cluster
    for sent in sentences:
        # print("In IDFs sent.getPreProWords() preproWords", sent.preproWords)
        # every word in a sentence
        for word in sent.getPreProWords():
            
            # not to calculate a word's IDF value more than once
            if sent.getWordFreq().get(word, 0) != 0:
                words[word] = words.get(word, 0)+ 1

    # for each word in words
    for word in words:
        n = words[word]
        
        # avoid zero division errors
        try:
            w2.append(n)
            idf = math.log10(float(N)/n)
        except ZeroDivisionError:
            idf = 0
                
        # reset variables
        idfs[word] = idf
            
    return idfs

def TF_IDF(sentences):
    # Method variables
    tfs = TFs(sentences)
    idfs = IDFs(sentences)
    retval = {}

    # for every word
    for word in tfs:
        #calculate every word's tf-idf score
        try:
            tf_idfs =  tfs[word] * idfs[word]
            # print("OK!", word)
        except KeyError:
        	print("No OK", word)
        
        # add word and its tf-idf score to dictionary
        if retval.get(tf_idfs, None) == None:
            retval[tf_idfs] = [word]
        else:
            retval[tf_idfs].append(word)

    return retval

def sentenceSim(sentence1, sentence2, IDF_w):
	numerator = 0
	denominator = 0	
	# print("sentence to compare",sentence1.originalWords, sentence2.originalWords)
	for word in sentence2.getPreProWords():		
		numerator+= sentence1.getWordFreq().get(word,0) * sentence2.getWordFreq().get(word,0) *  IDF_w.get(word,0) ** 2

	for word in sentence1.getPreProWords():
		denominator+= ( sentence1.getWordFreq().get(word,0) * IDF_w.get(word,0) ) ** 2

	# check for divide by zero cases and return back minimal similarity
	try:
		return numerator / math.sqrt(denominator)
	except ZeroDivisionError:
		# return float("-inf")
		return sentence_sim_exception

def buildQuery(sentences, TF_IDF_w, n):
	#sort in descending order of TF-IDF values
	scores = TF_IDF_w.keys()
	scores = sorted(scores, reverse=True)	
	
	i = 0
	j = 0
	queryWords = []

	# print("n, len(scores):", n, len(scores))
	# if len(scores) == 1:
	# 	print(TF_IDF_w)
	# select top n words
	while(i<n):
		words = TF_IDF_w[scores[j]]
		for word in words:
			queryWords.append(word)
			i=i+1
			if (i>n): 
				break
		j=j+1

	# return the top selected words as a sentence
	# return sentence.sentence("query", queryWords, queryWords)
	return Sentence(queryWords, queryWords)


def bestSentence(sentences, query, IDF):
	best_sentence = None
	maxVal = float("-inf")

	for sent in sentences:
		similarity = sentenceSim(sent, query, IDF)		

		if similarity > maxVal:
			best_sentence = sent
			maxVal = similarity
	# print("query", query.originalWords)
	# if best_sentence == None:
		# print([ sent.originalWords for sent in sentences ])
	global counter_sim_devide0
	# print("maxVal", maxVal)
	if maxVal == sentence_sim_exception:
		print("Some errors")
		counter_sim_devide0 += 1
	sentences.remove(best_sentence)

	return best_sentence

def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):	
	summary = [best_sentence]
	sum_len = len(best_sentence.getPreProWords())

	MMRval={}

	# keeping adding sentences until number of words exceeds summary length
	while (sum_len < summary_length and len(sentences) != 0):	
		MMRval={}		

		for sent in sentences:
			MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)

		maxxer = max(MMRval, key=MMRval.get)
		summary.append(maxxer)
		sentences.remove(maxxer)
		sum_len += len(maxxer.getPreProWords())	

	return summary

def MMRScore(Si, query, Sj, lambta, IDF):	
	Sim1 = sentenceSim(Si, query, IDF)
	l_expr = lambta * Sim1
	value = [float("-inf")]

	for sent in Sj:
		Sim2 = sentenceSim(Si, sent, IDF)
		value.append(Sim2)

	r_expr = (1-lambta) * max(value)
	MMR_SCORE = l_expr - r_expr	

	# return MMRScore
	return l_expr - r_expr

if __name__=='__main__':
    data = read_json(pjoin(sys.argv[1], f"{sys.argv[2]}.json"))
    articles = []
    abstracts = []
    
    for item in data:
        articles.append(item['article'])
        abstracts.append([item['abstract']]) 
    
    list_of_summarization = []
    for i in range(len(articles)):
        sample = articles[i]
        sentences = processFile(sample)
        original_sentences = [ sent.originalWords for sent in sentences ]
        IDF_w 		= IDFs(sentences)
        TF_IDF_w 	= TF_IDF(sentences)
        # build query; set the number of words to include in our query
        query = buildQuery(sentences, TF_IDF_w, 8)	
        # pick a sentence that best matches the query	
        best1sentence = bestSentence(sentences, query, IDF_w)
        summary = makeSummary(sentences, best1sentence, query, 100, 0.5, IDF_w)
        tmp_summary = []
        for sent in summary:
            original_sent = sent.getOriginalWords()
            tmp_summary.append(original_sent)
        
        list_of_summarization.append(tmp_summary)
    
    test_rouge(list_of_summarization, abstracts, 8)

	