
# Gerber_section ----------------------------------------------------------



library(XML)
library(tm)

##### Constructing TF-IDF Matrices #####

# read some news data from an XML file and transform it into a corpus. the following
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





# modeling_section 1 --------------------------------------------------------

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
  
# make model
library(randomForest)
my_mod <- randomForest(as.factor(sentiment) ~ .,
                       data=training_set_reduced, 
                       importance=TRUE, 
                       ntree=2000)
my_mod

# make predictions and write to file
preds <- predict(my_mod, newdata = vars_test_reduced)
preds

id_vec <- 1:length(preds)

# create tibble for output
output <- as_tibble(cbind(id_vec, preds))
colnames(output) <- c("id","sentiment") # rename cols

# write final output to file
write.csv(output, file = "khg3je_submission1.csv", row.names = FALSE)


# ------------------# 
#        KNN        # 
# ------------------# 
euclideanDist <- function(a, b){
     d = 0
     for(i in c(1:(length(a)-1) ))
       {
           d = d + (a[[i]]-b[[i]])^2
         }
     d = sqrt(d)
     return(d)
   }
####Resource: http://dataaspirant.com/2017/01/02/k-nearest-neighbor-classifier-implementation-r-scratch/
knn_predict <- function(test_data, train_data, k_value){
    pred <- c()  #empty pred vector 
    #LOOP-1
    for(i in c(1:nrow(test_data))){
      #looping over each record of test data
      eu_dist =c()          #eu_dist & eu_char empty  vector
      eu_char = c()
      five = 0              #1-5 sensitiment variable initialization with 0 value
      four = 0
      three = 0
      two = 0
      one = 0
      #LOOP-2-looping over train data 
    for(j in c(1:nrow(train_data))){
      
    #adding euclidean distance b/w test data point and train data to eu_dist vector
      eu_dist <- c(eu_dist, euclideanDist(test_data[i,], train_data[j,]))
    #adding class variable of training data in eu_char
      eu_char <- c(eu_char, as.character(train_data[j,][[83]])) # column #83 contains sentiment scores
        }
     eu <- data.frame(eu_char, eu_dist) #eu dataframe created with eu_char & eu_dist columns
     eu <- eu[order(eu$eu_dist),]       #sorting eu dataframe to gettop K neighbors
     eu <- eu[1:k_value,]               #eu dataframe with top K neighbors
    #Loop 3: loops over eu and counts classes of neibhors.
    for(k in c(1:nrow(eu))){
      if(as.character(eu[k,"eu_char"]) == "5"){
         five = five + 1
            }
      if(as.character(eu[k,"eu_char"])=="4"){
         four = four+1
           }
      if(as.character(eu[k,"eu_char"])=="3"){
         three = three+1
            }
      if(as.character(eu[k,"eu_char"])=="2"){
         two = two+1
          }
      else
         one= one + 1
          }
      pred <- c(pred, max(c(five,four,three,two,one))) 
           }
     return(pred) #return pred vector
}

K = 5
predictions <- knn_predict(vars_test_reduced, training_set_reduced, K) #calling knn_predict()
predictions
test = read.csv("test.csv")
table = data.frame(test$id,predictions)
write.table(table,file="knn_kaggle_3-4.csv",sep = ',', row.names = F,col.names = c('id','sentiment'))

