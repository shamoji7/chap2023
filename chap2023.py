import math
import pandas as pd
import re
from janome.tokenizer import Tokenizer
from asari.api import Sonar
import os
import sys
from IPython.terminal.prompts import Token


sonar = Sonar()
t = Tokenizer()


input = 'NBA ドラフト'
argv = input.split()
argc = len(argv)

def cos(argv):
    INDEX = "index"
    #index_file = INDEX + "/sample_index.txt"
    index_file = INDEX + "/index.txt"

    idf_scores = {}
    tfidf_scores = {}
    query_tf = {}
    query_tfidf = {}
    stopwords = {}
    query_words = {}
    ranking_docs = {}

    #tfidf_scoresからデータフレームを作成
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


    #クエリに対する処理
    #argv = '吾輩は猫である'
    query_file = 'query'

    pattern = re.compile(r"^[ -ー]$")
    stopwords['という'] = 1
    stopwords['にて'] = 1


    #tokens = t.tokenize(argv[1])
    tokens = []
    for i in range(argc):
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



    #クエリtfidf_scoresからクエリのデータフレームを作る
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

    #クエリが索引後辞書の中でどの文書名に属すのかを確認し、対象文書を同定
    for query_word in query_words:
        if query_word in tfidf_scores:
            for doc in tfidf_scores[query_word]:
                ranking_docs[doc] = 1

    #cos類似度計算
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

    return sorted(ranking_docs.items(), key=lambda x:x[1], reverse=True)
'''
        print('コマンドライン引数:', argc)
        print('分ち書き後クエリ:', flat_token)
        print('[docs, cos類似度]')
        for doc in sorted(ranking_docs.items(), key=lambda x:x[1], reverse=True):
            print(doc)
'''

def text_emotions():
    data = []
    for filename in os.listdir('text'):
        doc_box = []
        f = open('text/' + filename, 'r')
        for row in f:
            row = row.rstrip()
            doc_box.append(row)
        result_doc = ''.join(doc_box)
        res = sonar.ping(text=result_doc)
        classes = res['classes']
        positive_confidence = next((item['confidence'] for item in classes if item['class_name'] == 'positive'), 0)
        negative_confidence = next((item['confidence'] for item in classes if item['class_name'] == 'negative'), 0)

        scaled_confidence = (positive_confidence - negative_confidence)
        data.append(scaled_confidence)
    return data


def query_emotions(argv):
    query = ' '.join(argv)
    res = sonar.ping(text=query)
    classes = res['classes']
    positive_confidence = next((item['confidence'] for item in classes if item['class_name'] == 'positive'), 0)
    negative_confidence = next((item['confidence'] for item in classes if item['class_name'] == 'negative'), 0)
    scaled = (positive_confidence - negative_confidence)
    return scaled

cosinfo = cos(argv)
te = text_emotions()
qe = query_emotions(argv)

for row in cosinfo:
    print('cos similarity:', row)
for i in range(len(te)):
    print('text_emotions:', 'doc', i+1, te[i])

print('query_emotions:', qe)





