
library(XML)
library(tm)

##### Constructing TF-IDF Matrices #####

# read some news data from file and transform it into a corpus. the following
# data frame will have three columns:  id (document identifier), t (title), date (rough date)
# and c (content).
big_huge_func <- function(file_name) {
  
  document.data.frame = read.csv(file_name, stringsAsFactors = FALSE)
  
  str(document.data.frame)
  
  # for the purpose of this example, we only care about content.
  document.data.frame = as.data.frame(document.data.frame[,"text"], stringsAsFactors = FALSE)
  
  # convert this part of the data frame to a corpus object.
  news = VCorpus(DataframeSource(document.data.frame))
  
  # regular indexing returns a sub-corpus
  inspect(news[1:2])
  
  # double indexing accesses actual documents
  news[[1]]
  news[[1]]$content
  
  # compute TF-IDF matrix and inspect sparsity
  news.tfidf = DocumentTermMatrix(news, control = list(weighting = weightTfIdf))
  news.tfidf  # non-/sparse entries indicates how many of the DTM cells are non-zero and zero, respectively.
  # sparsity is number of non-zero cells divided by number of zero cells.
  
  # inspect sub-matrix:  first 5 documents and first 5 terms
  news.tfidf[1:5,1:5]
  as.matrix(news.tfidf[1:5,1:5])
  
  ##### Reducing Term Sparsity #####
  
  # there's a lot in the documents that we don't care about. clean up the corpus.
  news.clean = tm_map(news, stripWhitespace)                          # remove extra whitespace
  news.clean = tm_map(news.clean, removeNumbers)                      # remove numbers
  news.clean = tm_map(news.clean, removePunctuation)                  # remove punctuation
  news.clean = tm_map(news.clean, content_transformer(tolower))       # ignore case
  news.clean = tm_map(news.clean, removeWords, stopwords("english"))  # remove stop words
  news.clean = tm_map(news.clean, stemDocument)                       # stem all words
  
  # compare original content of document 1 with cleaned content
  news[[1]]$content
  news.clean[[1]]$content  # do we care about misspellings resulting from stemming?
  
  # recompute TF-IDF matrix
  news.clean.tfidf = DocumentTermMatrix(news.clean, control = list(weighting = weightTfIdf))
  
  # reinspect the first 5 documents and first 5 terms
  news.clean.tfidf[1:5,1:5]
  as.matrix(news.clean.tfidf[1:5,1:5])
  
  # we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
  tfidf.99 = removeSparseTerms(news.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
  tfidf.99
  as.matrix(tfidf.99[1:5,1:5])
  
  
  # which documents are most similar?
  dtm.tfidf.99 = as.matrix(tfidf.99)
  dtm.dist.matrix = as.matrix(dist(dtm.tfidf.99))
  most.similar.documents = order(dtm.dist.matrix[1,], decreasing = FALSE)
  news[[most.similar.documents[1]]]$content
  news[[most.similar.documents[2]]]$content
  news[[most.similar.documents[3]]]$content
  news[[most.similar.documents[4]]]$content
  news[[most.similar.documents[5]]]$content
  
  tfidf.99
  
}





# Additional Data Prep ------

library(tidyverse)

train.99 <- big_huge_func("train.csv")

# make train
vars_train <- as_tibble(as.matrix(train.99[1:981,1:126]))
sentiment_train <- read_csv("train.csv")[,"sentiment"]
training_set <- cbind(vars_train, sentiment_train)

# make test
test.99 <- big_huge_func("test.csv")
vars_test <- as_tibble(as.matrix(test.99[1:979,1:132]))

# remove variables from train-and-test that are not in both
train_names <- names(training_set)
test_names <- names(vars_test)
intersect_names <- intersect(train_names, test_names)

training_set_reduced <- training_set %>% 
  select(one_of(intersect_names),
         sentiment)

vars_test_reduced <- vars_test %>%
  select(one_of(intersect_names))


# more fixing names
colnames(training_set_reduced)[40] <- "next_FIXED"
colnames(vars_test_reduced)[40] <- "next_FIXED"

training_set_reduced
  

# KNN ------------------------------------------------

# load modes library
library(modes)

# declare KNN_predict function
#
# inputs: single test observation, all training observations, value for 'k'
# outputs: predicted class for test_observation
KNN_predict <- function(test_observation, training_data, k_value) {
  
  # declare calc_eulic function
  #
  # inputs: single training observation, single test_observation
  # outputs: euclidean distance between two observations
  calc_euclid <- function(training_observation, test_observation){
    
    sqrt(sum((training_observation - test_observation)^2))
    
  }
  
  # calculate all distances between single test observation and each training observation
  distances <- apply(X = training_data[-83], MARGIN = 1, FUN = calc_euclid,
                     test_observation=test_observation)
  
  distances_df <- as_tibble(cbind(distances, training_data["sentiment"]))
  
  k_nearest_classes <- distances_df %>%
    arrange(distances) %>%
    slice(1:k_value) %>%
    pull(sentiment)
  
  # round median of mode(s) of k_nearest_classes
  predicted_class <- round(median(unname(modes::modes(k_nearest_classes)[1,])))
  
}

# calculate all class predictions for test_data
preds <- apply(X = vars_test_reduced, MARGIN = 1, FUN = KNN_predict
               , training_data = training_set_reduced, k_value = 5)


id_vec <- 1:length(preds)

# create tibble for output
output <- as_tibble(cbind(id_vec, preds))
colnames(output) <- c("id","sentiment") # rename cols

# write final output to file
write.csv(output, file = "khg3je_KNN.csv", row.names = FALSE)


