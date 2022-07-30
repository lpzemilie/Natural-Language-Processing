#   A1: NPL Case Analyzing VC Press Releases
#   Student: Lopez Émilie
#   Last updated: Feb 1, 2022



#   1. PREPARING ENVIRONMENT

#   Setting working directory
setwd("~/Desktop/Text Analytics/Hult_NLP_student_intensive/assignments/A1 NLP Case Analyzing VC Press Releases")

#   Loading library
library(text2vec)
library(caret)
library(tm)
library(glmnet)
library(qdap)
library(wordcloud)
library(tidyr) 
library(tidytext)
library(lexicon) 
library(echarts4r) 
library(corpus) 
library(dplyr)
library(NLP)
library(RColorBrewer)
library(RCurl)
library(plotrix)
library(ggplot2)
library(ggthemes)
library(ggalt)

#   Reading files
training <- read.csv('training_VC_PressRelease_software_seriesA.csv')
testing <- read.csv('unlabeled_VC_PressRelease_software_seriesA.csv')

#   Loading custom cleaning function

#   Clean function for model
clean <-function(x){
  x <- removePunctuation(x)
  x <- stripWhitespace(x)
  x <- tolower(x)
  return(x)
}

#   Try to lower function
trytolower <- function(x){
  y <- NA 
  tryError <- tryCatch(tolower(x), error = function(e) e) 
  if (!inherits(tryError, 'error'))   
    y <- tolower(x) 
  return(y) 
}

#   Clean corpus function
cleancorpus<-function(corpus, customstopwords){
  corpus <- tm_map(corpus, content_transformer(qdapRegex::rm_url))
  corpus <- tm_map(corpus, content_transformer(replace_contraction)) 
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(trytolower))
  corpus <- tm_map(corpus, removeWords, customstopwords)
  return(corpus)
}

#   Bigram tokens functions
bigramTokens <-function(x){
  unlist(lapply(NLP::ngrams(words(x), 2), paste, collapse = " "), 
         use.names = FALSE)
}

#   Stop words
stopwords <- c(stopwords('SMART'), 'company', 'companies', 'million', 'venture capital', 'venture', 'ventures', 'series', 'funding', 'investment', 'round', 'announced')



#   2. LOOKING AT THE DATA

#   Labeled data
summary(training)
names(training)
head(training,1)   
table(training$newsProvider)   # Note: Press releases all comes from prnewswire.com
table(training$y_softwareAI)  # Note: 372 press releases are related to software, AI or machine learning.

#   Unlabeled data
summary(testing)
names(testing)
head(testing,1)
table(testing$newsProvider)  
table(testing$y_softwareAI) # Note: 192 press releases are related to software, AI or machine learning.

#   Note: Both files are made with the same eight variables. 
#   Note: The character variables prHeadline, headlineTxt and completeTxt are the most interesting to build a model and understand patterns. 



#   3. BUILDING A SUPERVISED MODEL

#   Combining headline, headline text and complete text in a new variable "text"
training$text <- as.character(paste(training$prHeadline, training$headlineTxt, training$completeTxt, sep = ' '))
testing$text <- as.character(paste(testing$prHeadline, testing$headlineTxt, testing$completeTxt, sep = ' '))

#   Cleaning the data with "clean" custom function
training$text <- clean(training$text)
testing$text <- clean(testing$text)

#   Creating vocabulary from tokenization
training_iterator <- itoken(training$text, preprocess_function = list(tolower))
training_text <- create_vocabulary(training_iterator, stopwords = stopwords('SMART'))
nrow(training_text)   # Note: There are 48,542 tokens (which is too much)

#   Keeping only relevant keywords by pruning the vocabulary
training_text <- prune_vocabulary(training_text, term_count_min = 10, doc_proportion_max = 0.5, doc_proportion_min = 0.001)
nrow(training_text)   # Note: The number of tokens has been reduced to 5,779.

#   Creating a document term matrix
vectorizer <- vocab_vectorizer(training_text)
training_dtm <- create_dtm(training_iterator, vectorizer)

#   Building a model from training data
model <- cv.glmnet(training_dtm,
                     y = as.factor(training$y_softwareAI),
                     alpha = 0.9,
                     family = 'binomial',
                     type.measure = 'auc',
                     nfolds = 5,
                     intercept = F)

# Keeping only impacting terms
head(coefficients(model),10)    # Note: Coefficients are all zero and need to be removed
best_terms <- subset(as.matrix(coefficients(model)), as.matrix(coefficients(model)) !=0)  # Note: Zero coefficients removed
best_terms  # Note: artificial, machine, intelligence, developing and software are the most impacting term in the model 

# Making training predictions
training_prediction <- predict(model, training_dtm, type = 'class')
confusionMatrix(as.factor(training_prediction),  
                as.factor(training$y_softwareAI))  # Note: The model is 86.7% accurate.



#   4. TESTING THE MODEL

# Creating vocabulary from tokenization
testing_iterator <- itoken(testing$text, preprocess_function = list(tolower),
                   preprocess_function = list(tolower))

# Using the original vectorizer function to retain the original terms 
testing_dtm <- create_dtm(testing_iterator, vectorizer)

# Making testing predictions
testing_prediction <- predict(model, testing_dtm, type = 'class')
confusionMatrix(as.factor(testing_prediction), 
                as.factor((testing$y_softwareAI)))   # Note: The model is close to 86.5% accurate. The model is stable.



#  OPTIONAL: CLEANING ENVIRONMENT TO RELEASE SOME RAM
rm(best_terms, model, testing_dtm, testing_prediction, training_dtm, training_prediction, training_text, testing_iterator, training_iterator, vectorizer, clean)
gc()




#   5. ANALYZING TEXT CLASSIFIED AS SOFTWARE, AI OR MACHINE LEARNING RELATED

#   Combine press releases classified as software, AI or machine learning related.
txt <- rbind(training[grepl("TRUE", training$y_softwareAI), ], testing[grepl("TRUE", testing$y_softwareAI), ])

#   Creating a corpus
txt$doc_id <-  1:nrow(txt)
txt_corpus <- data.frame(doc_id = txt$doc_id, text = txt$text)
txt_corpus <- VCorpus(DataframeSource(txt_corpus))

#   Cleaning corpus
txt_corpus <- cleancorpus(txt_corpus, stopwords)


#   A. UNIGRAM TOKENS WORDCLOUD

#   Creating a document term matrix
txt_tdm <- TermDocumentMatrix(txt_corpus)
txt_tdmm <- as.matrix(txt_tdm)

#   Creating a frequency table
txt_frequency <- sort(rowSums(txt_tdmm), decreasing = TRUE)
txt_frequency <- data.frame(word = names(txt_frequency), freq = txt_frequency)
head(txt_frequency,10)  

#   Creating a color palette
txt_pal <- brewer.pal(8, "Reds")
txt_pal <- txt_pal[-(1:2)]

#   Creating a wordcloud
wordcloud(txt_frequency$word,
          txt_frequency$freq,
          random.order = FALSE,
          max.words    = 30,
          colors = txt_pal)


#   B. BIGRAM TOKENS WORDCLOUD

#   Creating a document term matrix
txt_tdm_bi  <- TermDocumentMatrix(txt_corpus, control=list(tokenize=bigramTokens))
txt_tdmm_bi <- as.matrix(txt_tdm_bi)

#   Creating a frequency table
txt_frequency_bi <- sort(rowSums(txt_tdmm_bi), decreasing = TRUE)
txt_frequency_bi <- data.frame(word = names(txt_frequency_bi), freq = txt_frequency_bi)
head(txt_frequency_bi)   # Note: the first term is not text and could not be cleaned with the cleancorpus function
txt_frequency_bi <- txt_frequency_bi[2:nrow(txt_frequency_bi),]   # Note: first term removed
head(txt_frequency_bi)   

#   Creating a wordcloud
wordcloud(txt_frequency_bi$word,
          txt_frequency_bi$freq,
          random.order = FALSE,
          max.words    = 10,
          colors = txt_pal,
          scale = c(2,0.5))



#   C. POLARITY ANALYSIS

#   Tidying the document term matrix
tidy_txt <- tidy(txt_tdm)

#   Getting bing lexicon
bing <- get_sentiments(lexicon = c("bing"))

#   Performing an inner join
polarity_txt <- inner_join(tidy_txt, bing, by=c ('term'='word'))

#   Looking at polarity
aggregate(count~sentiment, polarity_txt, sum)  # Note: positive polarity
17252/(5563+17252)  # 75% positive



#   D. SENTIMENT ANALYSIS 

#   Getting nrc lexicon
setwd("~/Desktop/Text Analytics/Hult_NLP_student_intensive/lessons/class3/data")
nrc <- read.csv('nrcSentimentLexicon.csv')
nrc <- nrc_emotions

#   Cleaning lexicon
nrc <- nrc %>% pivot_longer(-term, names_to = "emotion", values_to = "freq")
nrc <- subset(nrc, nrc$freq>0 )
nrc$freq <- NULL 

#   Performing inner join
sentiment_txt <- inner_join(tidy_txt,nrc, by=c('term' = 'term'))

#   Creating a radar chart
emotions_txt <- data.frame(table(sentiment_txt$emotion))
names(emotions_txt) <- c('emotion', 'termsCt')
emotions_txt %>% 
  e_charts(emotion) %>% 
  e_radar(termsCt, max = max(emotions_txt$termsCt), name = "Press releases emotions") %>%
  e_tooltip(trigger = "item")%>%
  e_theme("red")  # Note: Trust is the main emotions coming from the press realeases

# Investigating trust vocabulary
trust <- sentiment_txt %>% filter(emotion == "trust")
trust <- aggregate(count~term, trust, sum) 
trust <- trust[order(trust$count, decreasing=T),]
head(trust)   # Note: the words team, management, leading, income, related and cash are the vocabulary of trust the most present in press releases.



#  OPTIONAL: CLEANING ENVIRONMENT TO RELEASE SOME RAM
rm(bing, emotions_txt, polarity_txt, testing, training, txt, trust, sentiment_txt, tidy_txt, txt_corpus, txt_frequency, txt_frequency_bi, txt_tdm, txt_tdm_bi, txt_tdmm, txt_tdmm_bi, txt_pal, bigramTokens)
gc()


#   E. ANALYZING AMOUNT OF INVESTMENT

#   Note: the four first steps are here to have unclean text with punctuation, which can impact the amounts (7.3 != 73 ).

#   Setting working directory
setwd("~/Desktop/Text Analytics/Hult_NLP_student_intensive/assignments/A1 NLP Case Analyzing VC Press Releases")

#   Reading files
training <- read.csv('training_VC_PressRelease_software_seriesA.csv')
testing <- read.csv('unlabeled_VC_PressRelease_software_seriesA.csv')

#   Combining headline, headline text and complete text in a new variable "text"
training$text <- as.character(paste(training$prHeadline, training$headlineTxt, training$completeTxt, sep = ' '))
testing$text <- as.character(paste(testing$prHeadline, testing$headlineTxt, testing$completeTxt, sep = ' '))

#   Combine press releases classified as software, AI or machine learning related.
txt <- rbind(training[grepl("TRUE", training$y_softwareAI), ], testing[grepl("TRUE", testing$y_softwareAI), ])

#  Extracting dollar amount 
allFunds <- list()
for(i in 1:nrow(txt)){
  print(i)
  tmpDollar <- gsub("\\$", "USD", txt$text[i]) 
  tmp <- regexpr("\\b(USD|GBP|EUR) *[0-9.]+", tmpDollar) 
  tmpDollar <- regmatches(tmpDollar, tmp) 
  value <- as.numeric(gsub("[^0-9.]", "", tmpDollar))
  value <- ifelse(length(value)==0, 'NA', value)
  currency <- gsub("[0-9. ]", "", tmpDollar)
  currency <- ifelse(length(currency)==0, 'NA', currency)
  allFunds[[i]] <- data.frame(currency = currency,
                              val      = value)
}

allFunds <- do.call(rbind, allFunds)
table(allFunds)  # Note: some amounts are in EUR, others in USD, some were not found "NA"

#   Removing NA
allFunds$val <- gsub("NA","",allFunds$val)

#   Replacing EUR by USD (1 EUR = 1.11816 USD)
grep("EUR",allFunds$currency)
allFunds$val[181] <- round(as.numeric(allFunds$val[181])*1.11816,1)
allFunds$val[227] <- round(as.numeric(allFunds$val[227])*1.11816,1)
allFunds$val[278] <- round(as.numeric(allFunds$val[278])*1.11816,1)
allFunds$val[468] <- round(as.numeric(allFunds$val[468])*1.11816,1)
allFunds$val[552] <- round(as.numeric(allFunds$val[552])*1.11816,1)

#   Basic statistics
summary(as.numeric(allFunds$val))   # Note: On average, the funding amount is USD 45.3 million

# Removing outliers 
value <- as.numeric(allFunds$val)
Q <- quantile(value, probs=c(.25, .75), na.rm = TRUE)
iqr <- IQR(value, na.rm=TRUE)
up <-  Q[2]+1.5*iqr  
low <- Q[1]-1.5*iqr 
value_subset <- subset(value, value > (Q[1] - 1.5*iqr) & value < (Q[2]+1.5*iqr))

# Creating an histogram
hist(value_subset, main = "Funding amount", col = "red")



#   F. SENTIMENT ANALYSIS BASED ON FUNDING AMOUNT MEDIAN

#   Split data depending on the median
value <- as.data.frame(value)
value <- value %>% mutate(split=ifelse(value>=17,T,F))

txt <- cbind(txt$text, value)
under_median <- txt[grepl("FALSE", value$split), ]
above_median <- txt[grepl("TRUE", value$split), ]

#   Creating a corpus 
under_median$doc_id <-  1:nrow(under_median)
under_corpus <- data.frame(doc_id = under_median$doc_id, text = under_median$`txt$text`)
under_corpus <- VCorpus(DataframeSource(under_corpus))

above_median$doc_id <-  1:nrow(above_median)
above_corpus <- data.frame(doc_id = above_median$doc_id, text = above_median$`txt$text`)
above_corpus <- VCorpus(DataframeSource(above_corpus))

#   Cleaning corpus 
under_corpus <- cleancorpus(under_corpus, stopwords)
above_corpus <- cleancorpus(above_corpus, stopwords)

#   Creating a document term matrix
under_tdm <- TermDocumentMatrix(under_corpus)
under_tdmm <- as.matrix(under_tdm)

above_tdm <- TermDocumentMatrix(above_corpus)
above_tdmm <- as.matrix(above_tdm)

#   Tidying the document term matrix
tidy_under <- tidy(under_tdm)
tidy_above <- tidy(above_tdm)

#   Performing inner join
sentiment_under <- inner_join(tidy_under,nrc, by=c('term' = 'term'))
sentiment_above <- inner_join(tidy_above,nrc, by=c('term' = 'term'))

#   Creating a radar chart
emotions_under <- data.frame(table(sentiment_under$emotion))
emotions_above <- data.frame(table(sentiment_above$emotion))
emotion_median <- data.frame(emotions_above, emotions_under)
names(emotion_median) <- c('emotion', 'termsCt', 'emotion2', 'termCt2')
emotion_median %>% 
  e_charts(emotion) %>% 
  e_radar(termsCt, max = max(emotion_median$termsCt), name = "above median" ) %>%
  e_radar(termCt2, max = max(emotion_median$termCt2), name = "under median") %>%
  e_tooltip(trigger = "item")%>%
  e_theme("infographic")

#   Note: No significant difference in emotions between bigger funding and smaller fundings

# Investigating fear vocabulary
fear <- sentiment_under %>% filter(emotion == "fear")
fear <- aggregate(count~term, fear, sum) 
fear <- fear[order(fear$count, decreasing=T),]
head(fear)

# Investigating sadness vocabulary
sadness <- sentiment_under %>% filter(emotion == "sadness")
sadness <- aggregate(count~term, sadness, sum) 
sadness <- sadness[order(sadness$count, decreasing=T),]
head(sadness)



#  OPTIONAL: CLEANING ENVIRONMENT TO RELEASE SOME RAM
rm(fear, sadness, emotion_median, emotions_above, emotions_under, sentiment_above, sentiment_under, tidy_above, tidy_under, training, testing, nrc, txt, under_corpus, under_tdm, under_tdmm, value,allFunds, above_tdmm, above_tdm, above_corpus, currency, i, iqr, low, Q, tmp, tmpDollar, up, value_subset, bigramTokens, clean  )
gc()


#   G. TEXT ANALYSIS WITH MEDIAN

#   Creating a list
list_median <- list(above_median$`txt$text`,under_median$`txt$text`)

#  Creating a term document matrix
for(i in 1:length(list_median)){
  print(paste('working on',i, 'of', length(list_median)))
  tmp <- paste(list_median[[i]], collapse = ' ')
  tmp <- VCorpus(VectorSource(tmp))
  tmp <- cleancorpus(tmp, stopwords)
  tmp <- TermDocumentMatrix(tmp)
  list_median[[i]] <- as.matrix(tmp)
}

summary(list_median)

#   Merging based on row attributes
df <- merge(list_median[[1]], list_median[[2]], by ='row.names')

#   Calculating the absolute differences for  common terms
df$diff <- abs(df[,2] - df[,3])

#   Organize data frame for plotting
df<- df[order(df$diff, decreasing=TRUE), ]
top <- df[1:30, ]

#   Creating a pyramid plot
pyramid.plot(lx         = top[,2], 
             rx         = top[,3],    
             labels     = top[,1],  
             top.labels = c( names(top)[2], names(top)[1],  names(top)[3]), 
             gap        = 80, 
             main       = 'Words in Common',
             unit       = 'wordFreq') 



#   End of script





