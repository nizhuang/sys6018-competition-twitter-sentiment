library(XML)
library(tm)

##### Constructing TF-IDF Matrices #####

# read some news data from an XML file and transform it into a corpus. the following
# data frame will have three columns:  id (document identifier), t (title), date (rough date)
# and c (content).
document = read.csv("train.csv", stringsAsFactors = FALSE)
document = document[1:981,] 
document.data.frame = as.data.frame(document,stringsAsFactors = FALSE)
tweets = VCorpus(DataframeSource(document.data.frame))  # convert this part of the data frame to a corpus object.
inspect(tweets[1:2])  # regular indexing returns a sub-corpus

# double indexing accesses actual documents
tweets[[1]] 
tweets[[1]]$content

# compute TF-IDF matrix and inspect sparsity
tweets.tfidf = DocumentTermMatrix(tweets, control = list(weighting = weightTfIdf))
tweets.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
# sparsity is number of non-zero cells divided by number of zero cells.

# inspect sub-matrix:  first 5 documents and first 5 terms
tweets.tfidf[1:5,1:5]
as.matrix(tweets.tfidf[1:5,1:5])

##### Reducing Term Sparsity #####

# there's a lot in the documents that we don't care about. clean up the corpus.
tweets.clean = tm_map(tweets, stripWhitespace)                          # remove extra whitespace
tweets.clean = tm_map(tweets.clean, removeNumbers)                      # remove numbers
tweets.clean = tm_map(tweets.clean, removePunctuation)                  # remove punctuation
tweets.clean = tm_map(tweets.clean, content_transformer(tolower))       # ignore case
tweets.clean = tm_map(tweets.clean, removeWords, stopwords("english"))  # remove stop words
tweets.clean = tm_map(tweets.clean, stemDocument)                       # stem all words

# compare original content of document 1 with cleaned content
tweets[[1]]$content
tweets.clean[[1]]$content  # do we care about misspellings resulting from stemming?

# recompute TF-IDF matrix
tweets.clean.tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf))
# inspect(tweets.clean.tfidf[1:5, 1000:1005])

# reinspect the first 5 documents and first 5 terms
tweets.clean.tfidf[1:5,1:5]
# # inspect(tweets.clean.tfidf[1:5,1:5])
# as.matrix(tweets.clean.tfidf[1:5,1:5])

#####################################################

freq <- sort(colSums(as.matrix(tweets.clean.tfidf)),decreasing=TRUE)
wf <- data.frame(word=names(freq),freq=freq)
freq[1:25]
wf[1]

findFreqTerms(tweets.clean.tfidf, lowfreq=15)

library(ggplot2)  # frequency graph of words used at least 15 times
freqgraph <- ggplot(subset(wf,freq>15),aes(word,freq))
freqgraph <- freqgraph + geom_bar(stat="identity")
freqgraph <- freqgraph+theme(axis.text.x=element_text(angle=45, hjust=1))
freqgraph

library(wordcloud)
wordcloud(names(freq), freq, min.freq=8, color=brewer.pal(6, "Dark2"))


######################################################


# # we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
# tfidf.99 = removeSparseTerms(tweets.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
# tfidf.99
# as.matrix(tfidf.99[1:5,1:5])
# 
# tfidf.70 = removeSparseTerms(tweets.clean.tfidf, 0.70)  # remove terms that are absent from at least 70% of documents
# tfidf.70
# as.matrix(tfidf.70[1:5, 1:5])
# tweets.clean[[1]]$content
# 
# 
# # which documents are most similar?
# dtm.tfidf.99 = as.matrix(tfidf.99)
# dtm.dist.matrix = as.matrix(dist(dtm.tfidf.99))
# most.similar.documents = order(dtm.dist.matrix[1,], decreasing = FALSE)
# tweets[[most.similar.documents[1]]]$content
# tweets[[most.similar.documents[2]]]$content
# tweets[[most.similar.documents[3]]]$content
# tweets[[most.similar.documents[4]]]$content
# tweets[[most.similar.documents[5]]]$content

