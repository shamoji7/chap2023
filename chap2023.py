from IPython.terminal.prompts import Token
import math
import pandas as pd
import re
from janome.tokenizer import Tokenizer
import sys

t = Tokenizer()
argv = sys.argv
argc = len(argv)

INDEX = "index"
index_file = INDEX + "/index3.txt"

idf_scores = {}
tfidf_scores = {}
query_tf = {}
query_tfidf = {}
stopwords = {}
query_words = {}
ranking_docs = {}

f = open(index_file, 'r')

for line in f:
    line = line.rstrip()
    split_line = line.split('\t')
    word = split_line[0]
    doc = split_line[1]
    idf = float(split_line[2])
    tfidf = float(split_line[3])

    if word not in tfidf_scores:
        tfidf_scores[word] = {}
        tfidf_scores[word][doc] = tfidf
    else:
        tfidf_scores[word][doc] = tfidf

    if word not in idf_scores:
        idf_scores[word] = {}
        idf_scores[word] = idf
    else:
        idf_scores[word] = idf

tfidf_table = pd.DataFrame(tfidf_scores)
tfidf_table = tfidf_table.fillna(0)

#argv = '吾輩は猫である'
query_file = 'query'

pattern = re.compile(r"^[ -ー]$")
stopwords['という'] = 1
stopwords['にて'] = 1


#tokens = t.tokenize(argv[1])
tokens = []
for i in range(argc):
    if i != 0:
        list = t.tokenize(argv[i])
        tokens.append(list)
flat_t = [item for sublist in tokens for item in sublist]

flat_token = []
for token in flat_t:
    tmp = token.surface
    judge = pattern.match(tmp)
    flat_token.append(tmp)
    
    if judge:
        continue
    if tmp in stopwords:
        continue
        
    if tmp in query_words:
        query_words[tmp] += 1
    else:
        query_words[tmp] = 1




for index_word in idf_scores:
    query_tf[index_word] = {}
    query_tfidf[index_word] = {}
    query_tf[index_word][query_file] = 0
    query_tfidf[index_word][query_file] = 0

for query_word in query_words:
    if query_word in query_tf:
        query_tf[query_word][query_file] += 1
        
for query_word in query_words:
    if query_word in query_tf:
        query_tfidf[query_word][query_file] = float(query_tf[query_word][query_file]) * float(idf_scores[query_word]) 

query_table = pd.DataFrame(query_tfidf)

for query_word in query_words:
    if query_word in tfidf_scores:
        for doc in tfidf_scores[query_word]:
            ranking_docs[doc] = 1

for doc in ranking_docs:
    numerator = 0
    doc_vec = tfidf_table.loc[doc]
    query_vec = query_table.loc[query_file]
    for i in range(len(query_vec.values)):
        i_value = query_vec.values[i] * doc_vec.values[i]
        numerator = numerator + i_value

    denominator = 0
    query_value = 0
    doc_value = 0
    for i in range(len(query_vec.values)):
        query_value += query_vec.values[i]**2
        doc_value += doc_vec.values[i]**2
    
    denominator = math.sqrt(query_value) * math.sqrt(doc_value)
    cosine = numerator / denominator
    
    ranking_docs[doc] = cosine

print('コマンドライン引数:', argc)
print('分ち書き後クエリ:', flat_token)
print('[docs, cos類似度]')
for doc in sorted(ranking_docs.items(), key=lambda x:x[1], reverse=True):
    print(doc)