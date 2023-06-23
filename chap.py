import re
import math
import os
from janome.tokenizer import Tokenizer

t = Tokenizer()

DATA =  "text"
INDEX = "index"
index_file = INDEX + "/index3.txt"

index_words = {}
stopwords = {}
docs = {}
df = {}
idf = {}

pattern = re.compile(r"^[ -ー]$")

stopwords['という'] = 1
stopwords['にて'] = 1


for filename in os.listdir(DATA):
    f = open(DATA + '/' + filename, 'r')
    for line in f:
        tokens = t.tokenize(line)
        for token in tokens:
            tmp = token.surface
            judge = pattern.match(tmp)
            if judge:
                continue
            if tmp in stopwords:
                continue

            if tmp in index_words:
                if filename in index_words[tmp]:
                    index_words[tmp][filename] += 1
                else:
                    index_words[tmp][filename] = 1

            else:
                index_words[tmp] = {}
                index_words[tmp][filename] = 1

for word in index_words:
    for doc in index_words[word]:
        if doc not in docs:
            docs[doc] = 1

docs_size = len(docs)

for word in index_words:
    df[word] = len(index_words[word])

for word in df:
    idf[word] = math.log( ( docs_size / df[word] ) + 1 )

f2 = open(index_file, 'w')

for word in sorted(index_words):
    for doc in sorted(index_words[word]):
        tfidf = index_words[word][doc] * idf[word]
        #print(word + '\t' + doc + '\t' + str(idf[word]) + '\t' + str(tfidf) + '\n') #確認用
        f2.write(word + '\t' + doc + '\t' + str(idf[word]) + '\t' + str(tfidf) + '\n')

f.close()
f2.close()