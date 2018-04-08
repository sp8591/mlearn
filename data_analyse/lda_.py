doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
result = []
tokenizer = RegexpTokenizer(r'\w+')
for doc in doc_set:
    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)
    stop_words = ['a', 'an', 'the', 'I', 'to', 'is']
    stopped_tokens = [i for i in tokens if not i in stop_words]
    p_stemmer = PorterStemmer()
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    result.append(stemmed_tokens)
print result

from gensim import corpora, models, similarities

dictionary = corpora.Dictionary(result)
corpus = [dictionary.doc2bow(i) for i in result]
print dictionary
print dictionary.token2id
print corpus
tfidf = models.TfidfModel(corpus=corpus)
corpus_tfidf = tfidf[corpus]
print corpus_tfidf[1]
ldamodel = models.LdaModel(corpus=corpus, num_topics=3, id2word=dictionary, passes=20)

print ldamodel.print_topics(num_topics=3, num_words=4)

for i in ldamodel.get_document_topics(corpus):
    print i

sim = similarities.MatrixSimilarity(list(corpus_tfidf))
print list(sim)

