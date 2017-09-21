library(XML)
library(class)
library(tm)
library(glmnet)
setwd("~/Documents/sys6018-competition-twitter-sentiment-master")
# read some news data from an XML file and transform it into a corpus. the following
# data frame will have three columns:  id (document identifier), t (title), date (rough date)
# and c (content).
document = read.csv("train.csv",header = T, stringsAsFactors = FALSE)
document = document[1:979,]
length(document)
document.data.frame = as.data.frame(document,stringsAsFactors = FALSE)
tweets = VCorpus(DataframeSource(document.data.frame))

# compute TF-IDF matrix and inspect sparsity
tweets.tfidf = DocumentTermMatrix(tweets, control = list(weighting = weightTfIdf))
tweets.tfidf # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
# sparsity is number of non-zero cells divided by number of zero cells.

# there's a lot in the documents that we don't care about. clean up the corpus.
tweets.clean = tm_map(tweets, stripWhitespace)                          # remove extra whitespace
tweets.clean = tm_map(tweets.clean, removeNumbers)                      # remove numbers
tweets.clean = tm_map(tweets.clean, removePunctuation)                  # remove punctuation
tweets.clean = tm_map(tweets.clean, content_transformer(tolower))       # ignore case
tweets.clean = tm_map(tweets.clean, removeWords, stopwords("english"))  # remove stop words
tweets.clean = tm_map(tweets.clean, stemDocument)                       # stem all words

# recompute TF-IDF matrix
tweets.clean.tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTf))
tweets.clean.tfidf

# we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
tfidf.99 = removeSparseTerms(tweets.clean.tfidf, 0.99)  # remove terms that are absent from at least 98% of documents (keep most terms)
tfidf.99
train.x = as.matrix(tfidf.99[,1:126])


############
# TEST DATA 
############
test = read.csv("test.csv",header = T,stringsAsFactors = FALSE)

test = VCorpus(DataframeSource(test))

# compute TF-IDF matrix and inspect sparsity
#test.tfidf = DocumentTermMatrix(test, control = list(weighting = weightTfIdf))
#test.tfidf 

# there's a lot in the documents that we don't care about. clean up the corpus.
test.clean = tm_map(test, stripWhitespace)                          # remove extra whitespace
test.clean = tm_map(test.clean, removeNumbers)                      # remove numbers
test.clean = tm_map(test.clean, removePunctuation)                  # remove punctuation
test.clean = tm_map(test.clean, content_transformer(tolower))       # ignore case
test.clean = tm_map(test.clean, removeWords, stopwords("english"))  # remove stop words
test.clean = tm_map(test.clean, stemDocument)                       # stem all words

test.clean[[1]]$content # test

# recompute TF-IDF matrix
test.clean.tfidf = DocumentTermMatrix(test.clean, control = list(weighting = weightTf(test)))
test.clean.tfidf


# we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
tfidf_99 = removeSparseTerms(test.clean.tfidf, 0.99)  # remove terms that are absent from at least 98% of documents (keep most terms)
tfidf_99
test.x = as.matrix(tfidf_99[,1:132])
test.x
data_test = as.data.frame(test.x)

################################################################################# 
# Parametric Model
# try fitting the model with all explanatory variables 
data_train = cbind(train.x,document$sentiment)
data_train = as.data.frame(data_train)
attach(data_train)
names(data_train)[127]<-paste("sentiment")

require("dplyr")
library(tidyr)
# remove variables from train-and-test that are not in both
train_names <- names(data_train)
test_names <- names(data_test)
intersect_names <- intersect(train_names, test_names)

training_set_reduced <- data_train %>% 
  select(one_of(intersect_names),
         sentiment)

vars_test_reduced <- data_test %>%
  select(one_of(intersect_names))


# model fit with all 82 variables 
model= glm(sentiment~., data = training_set_reduced)
summary(model)

par_pred_model = predict(model, newdata = data_test, type="response")


# refine the model 
# select the best model 
start<-glm(sentiment ~1,data= training_set_reduced)
end<-glm(sentiment~.,data= training_set_reduced)
result.s<-step(start, scope=list(upper=end), direction="both",trace=FALSE) 
summary(result.s)
anova(result.s)

par_pred_resutl.s = predict(result.s, newdata = data_test, type="response")


# Cross Validation 
library(boot)
# for model 
cv.err=cv.glm(training_set_reduced, model)
cv.err$delta  # 0.6184219 0.6183648
# for result.s
cv.err=cv.glm(training_set_reduced, result.s)
cv.err$delta # 0.5658330 0.5658133



# multinom
library(nnet)
fit_multinom= multinom(sentiment~.,training_set_reduced)
summary(fit_multinom)

#data_test = data.matrix(data_test)
par_pred = predict(fit_multinom,data_test, type="class")

# need cross validation for multinom for model comparison 

test = read.csv("test.csv")
table = data.frame(test$id,ceiling(par_pred))
write.table(table,file="par_kaggle_3-4.csv",sep = ',', row.names = F,col.names = c('id','sentiment'))







