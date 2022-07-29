#   A2: NPL Visualizations & EDA
#   Student: Lopez Émilie
#   Date: Jan 24, 2022



#   PREPARING ENVIRONMENT

#   Setting working directory
setwd("~/Desktop/Text Analytics/Hult_NLP_student_intensive/assignments/A2 NLP Visualizations EDA")

#   Loading libraries
library(dplyr)
library(tm)
library(tidytext)
library(qdap)
library(NLP)
library(RColorBrewer)
library(RCurl)
library(wordcloud)
library(lexicon) 
library(echarts4r) 
library(tidyr) 
library(corpus) 

#   Reading files
starbucks <- read.csv('sbux.csv')
dunkin <-read.csv('dunks.csv')

#   Loading custom functions

# Try to lower function
trytolower <- function(x){
  y <- NA 
  tryError <- tryCatch(tolower(x), error = function(e) e) 
  if (!inherits(tryError, 'error'))   
    y <- tolower(x) 
  return(y) 
}

# Stopwords
stopword_starbucks <- c(stopwords('SMART'),'starbucks','coffee')
stopword_dunkin <- c(stopwords('SMART'),'dunkin','donuts','donut','coffee')
stops <- c(stopwords('SMART'),'starbucks','dunkin','donuts','donut','coffee')
 
# Clean function
clean <-function(corpus, customStopwords){
  corpus <- tm_map(corpus, content_transformer(qdapRegex::rm_url)) 
  corpus <- tm_map(corpus, content_transformer(trytolower)) 
  corpus <- tm_map(corpus, removeWords, customStopwords)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}



#   LOOKING AT THE DATA

#   A. STARBUCKS
names(starbucks)
head(starbucks$text,5)  # Note: the second tweet is not in English.
summary(starbucks)

#   B. DUNKIN
names(dunkin)
head(dunkin$text,5)
summary(dunkin)



#   CLEANING THE DATA

#   A. STARBUCKS

#   1. Keeping only the English tweets

table(starbucks$lang)
starbucks <- starbucks[grepl("en", starbucks$lang), ]   # Note: 217 tweets removed
217/989   # Note: 22% of tweets are not in English

#   2. Creating a corpus

# Creating a column "doc_id" 
starbucks$doc_id <- 1:nrow(starbucks)

# Creating a data frame with the correct format for a corpus: first column "doc_id", second column "text"
starbucks_corpus <- data.frame(doc_id = starbucks$doc_id, text = starbucks$text)

# Creating a volatile corpus
starbucks_corpus <- VCorpus(DataframeSource(starbucks_corpus))

# Cleaning the corpus with custom function "clean"
starbucks_corpus <- clean(starbucks_corpus, stopword_starbucks)


#   B. DUNKIN

#   1. Keeping only the English tweets

table(dunkin$lang)
dunkin <- dunkin[grepl("en", dunkin$lang), ]   # Note: 113 tweets removed
113/1000  #Note: 11% of tweets are not in English

#   2. Create a corpus 

# Creating a column "doc_id" 
dunkin$doc_id <- 1:nrow(dunkin)

# Creating a data frame with the correct format for a corpus: first column "doc_id", second column "text"
dunkin_corpus <- data.frame(doc_id = dunkin$doc_id, text = dunkin$text)

# Creating a volatile corpus
dunkin_corpus <- VCorpus(DataframeSource(dunkin_corpus))

# Cleaning the corpus with custom function "clean"
dunkin_corpus <- clean(dunkin_corpus, stopword_dunkin)



#   ANALYSIS

#   A. STARBUCKS 

#   1. Number of characters per tweet
mean(nchar(starbucks$text))   # Note: On average, tweets related to Starbucks are 168 characters long (excluding spaces)


#   2. Word frequency analysis 

# Creating a term document matrix
starbucks_tdm <- TermDocumentMatrix(starbucks_corpus)
starbucks_tdmm <- as.matrix(starbucks_tdm)

# Creating a frequency table
starbucks_frequency <- sort(rowSums(starbucks_tdmm), decreasing = TRUE)
starbucks_frequency <- data.frame(word = names(starbucks_frequency), freq = starbucks_frequency)
head(starbucks_frequency,10)   # Note: terms are related to coronavirus

# Using a palette similar to Starbucks logo colors
display.brewer.all()
starbucks_pal <- brewer.pal(8, "Greens")
starbucks_pal <- starbucks_pal[-(1:2)]

# Creating a wordcloud
wordcloud(starbucks_frequency$word,
          starbucks_frequency$freq,
          random.order = FALSE,
          max.words    = 30,
          colors       = starbucks_pal)

# Investigating "boycottstarbucks"
starbucks_associations <- findAssocs(starbucks_tdm, 'boycottstarbucks', 0.30)
starbucks_associations  # Note: The hashtag "boycottstarbucks" is associated with the word "laptop", "afghanistan", "joe", "americans", "honor", "rescinding", "sit" and "left"

boycottstarbucks <- starbucks[grepl("BoycottStarbucks", starbucks$text), ]
head(boycottstarbucks$text,5)   # Note: Some people are not happy about the new Starbucks employee vaccine policy


#   3. Polarity analysis

# Creating a document term matrix
starbucks_dtm <- DocumentTermMatrix(starbucks_corpus)

# Tidying the document term matrix
tidy_starbucks <- tidy(starbucks_dtm)

# Getting bing lexicon
bing <- get_sentiments(lexicon = c("bing"))

# Performing an inner join
polarity_starbucks <-inner_join(tidy_starbucks, 
                                bing,by=c('term'='word'))

aggregate(count~sentiment, polarity_starbucks, sum)  # Note: no strong polarity


#   4. Sentiment analysis 

# Getting nrc lexicon
setwd("~/Desktop/Text Analytics/Hult_NLP_student_intensive/lessons/class3/data")
nrc <- read.csv('nrcSentimentLexicon.csv')
nrc <- nrc_emotions

# Cleaning lexicon
nrc <- nrc %>% pivot_longer(-term, names_to = "emotion", values_to = "freq")
nrc <- subset(nrc, nrc$freq>0 )
nrc$freq <- NULL 

# Performing an inner join
sentiment_starbucks <- inner_join(tidy_starbucks,nrc, by=c('term' = 'term'))

# Creating a radar chart
emotions_starbucks <- data.frame(table(sentiment_starbucks$emotion))
names(emotions_starbucks) <- c('emotion', 'termsCt')
emotions_starbucks %>% 
  e_charts(emotion) %>% 
  e_radar(termsCt, max = max(emotions_starbucks$termsCt), name = "Starbucks tweets emotions") %>%
  e_tooltip(trigger = "item")%>%
  e_theme("green")    # Note: Trust and anticipation are the feeling the comes out the most from the tweets. The anger feeling is generated by the coronavirus employee policy


#   B. DUNKIN

#   1. Number of characters per tweet
mean(nchar(dunkin$text))  # Note: On average, tweets related to Dunkin are 121 characters long (excluding spaces)

#   2. Word frequency analysis 

# Creating a term document matrix
dunkin_tdm  <- TermDocumentMatrix(dunkin_corpus)
dunkin_tdmm <- as.matrix(dunkin_tdm)

# Creating a frequency table
dunkin_frequency <- sort(rowSums(dunkin_tdmm), decreasing = TRUE)
dunkin_frequency <- data.frame(word = names(dunkin_frequency), freq = dunkin_frequency)
head(dunkin_frequency,10)   # Note: terms are related to sb and their song 'bazinga'

# Using a color palette similar to Dunkin logo colors
dunkin_pal <- brewer.pal(8, "Oranges")
dunkin_pal <- dunkin_pal[-(1:2)]

# Creating a wordcloud
wordcloud(dunkin_frequency$word,
          dunkin_frequency$freq,
          random.order = FALSE,
          max.words    = 30,
          colors       = dunkin_pal)


#   3. Polarity analysis

# Creating a document term matrix
dunkin_dtm <- DocumentTermMatrix(dunkin_corpus)

# Tidying the document term matrix
tidy_dunkin <- tidy(dunkin_dtm)

# Performing an inner join
polarity_dunkin <-inner_join(tidy_dunkin, 
                                bing,by=c('term'='word'))

aggregate(count~sentiment, polarity_dunkin, sum)  # Note: positive polarity 


#   4. Sentiment analysis 

# Performing inner join
sentiment_dunkin <- inner_join(tidy_dunkin,nrc, by=c('term' = 'term'))

# Creating a radar chart
emotions_dunkin <- data.frame(table(sentiment_dunkin$emotion))
names(emotions_dunkin) <- c('emotion', 'termsCt')
emotions_dunkin %>% 
  e_charts(emotion) %>% 
  e_radar(termsCt, max = max(emotions_starbucks$termsCt), name = "Dunkin tweets emotions") %>%
  e_tooltip(trigger = "item")%>%
  e_theme("fruit")    # Note: Anticipation is the feeling that comes out the most in the tweets.

# Investigating the fear feeling
sentiment_dunkin$term[grep("fear",sentiment_dunkin$emotion)]  # Note: the word "buzz" is the biggest part of the fear score. It is wrongly badly connotated. 


# COMMONALITIES STARBUCKS AND DUNKIN

setwd("~/Desktop/Text Analytics/Hult_NLP_student_intensive/assignments/A2 NLP Visualizations EDA")

# Making a list of data frames
starbucks_dunkin <- list.files(pattern ='dunks|sbux')
starbucks_dunkin <- lapply(starbucks_dunkin, read.csv)

#  Apply steps to each list element
for(i in 1:length(starbucks_dunkin)){
  print(paste('working on',i, 'of', length(starbucks_dunkin)))
  tmp <- paste(starbucks_dunkin[[i]]$text, collapse = ' ')
  tmp <- VCorpus(VectorSource(tmp))
  tmp <- clean(tmp, stops)
  tmp <- sapply(tmp, content)
  starbucks_dunkin[[i]] <- tmp
}

# Combining the documents into a corpus 
starbucks_dunkin <- unlist(starbucks_dunkin)
starbucks_dunkin <- VCorpus((VectorSource(starbucks_dunkin)))

# Making a term document matrix 
ctrl <- list(weighting = weightTfIdf)
starbucks_dunkin_tdm  <- TermDocumentMatrix(starbucks_dunkin, control = ctrl)
starbucks_dunkin_tdmm <- as.matrix(starbucks_dunkin_tdm)

# Making a comparison cloud
comparison.cloud(starbucks_dunkin_tdmm, 
                 max.words=30, 
                 random.order=FALSE,
                 colors=brewer.pal(ncol(starbucks_dunkin_tdmm),"Dark2"),
                 scale=c(3,0.1))    # Note: The SB19 song is dominant. The musician "haechan" appears in orange instead of green. Other brands, like Carhartt, impacted by the covid mandate appear.


# End of script  
