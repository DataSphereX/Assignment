###########Creating DTM: ###############
imdb <- read.csv("training_data.csv", stringsAsFactors = FALSE)

names(imdb)

table(imdb$sentiment)

##-------------------------------------------------------
corpus <- Corpus(VectorSource(imdb$review))


##-------------------------------------------------------
wordcloud(corpus, colors=rainbow(7), max.words=50)


##-------------------------------------------------------
corpus <- tm_map(corpus, tolower)

corpus <- tm_map(corpus, removePunctuation)

stopwords("english")[1:10]

corpus <- tm_map(corpus, removeWords, c("and","the","film","movi", stopwords("english")))

corpus <- tm_map(corpus, stemDocument) 

##-------------------------------------------------------
frequencies <- DocumentTermMatrix(corpus)

##-------------------------------------------------------
inspect(frequencies[1000:1005, 505:515])

findFreqTerms(frequencies, lowfreq = 20)

##-------------------------------------------------------
sparse = removeSparseTerms(frequencies, 0.920)

##-------------------------------------------------------
imdbsparse = as.data.frame(as.matrix(sparse))

##-------------------------------------------------------
colnames(imdbsparse) = make.names(colnames(imdbsparse))

length(names(imdbsparse))

##-------------------------------------------------------
imdbsparse$sentiment <- imdb$sentiment

##-------------------------------------------------------
imdbCART = rpart(sentiment~., data=imdbsparse, method="class")
rpart.plot(imdbCART)
prp(imdbCART, extra=2)

##-------------------------------------------------------
imdbCARTpred <- predict(imdbCART, data=imdbsparse)
imdbCARTCM <- table("Actual"= imdb$sentiment, "prediction"=imdbCARTpred[,2] > 0.5)
imdbCARTCM

accuracyCART <- (imdbCARTCM[1]+imdbCARTCM[4])/sum(imdbCARTCM)
round(accuracyCART * 100, 2) 


recall <- imdbCARTCM[4] / ( imdbCARTCM[4] + imdbCARTCM[2])
precision <- imdbCARTCM[4] / (imdbCARTCM[4] + imdbCARTCM[3])
F1Score_CART <- ((2* recall *precision) / (recall +precision))
F1Score_CART


##----------------------------------------------------------------
imdbRF <- randomForest(sentiment~., data=imdbsparse,ntree=20)
varImpPlot(imdbRF)

imdbRFCM <- table("Actual"= imdbsparse$sentiment, "prediction"=imdbRF$predicted>0.5)
imdbRFCM

accuracyRF <- (imdbRFCM[1]+imdbRFCM[4])/sum(imdbRFCM)
round(accuracyRF * 100, 2)

recall <- imdbRFCM[4] / ( imdbRFCM[4] + imdbRFCM[2])
precision <- imdbRFCM[4] / (imdbRFCM[4] + imdbRFCM[3])
F1Score_RF <- ((2* recall *precision) / (recall +precision))
F1Score_RF


##-----------------------------------------------------
imdblr <- glm(sentiment~., data = imdbsparse, family = "binomial")
imdblrCM <- table("Actual" = imdbsparse$sentiment, "Prediction" = imdblr$fitted.values > 0.5)
summary(imdblr)

##------------------------------------------------------
round((1 - (imdblr$deviance/imdblr$null.deviance))*100, 2)


words <- as.data.frame((coef(summary(imdblr))[coef(summary(imdblr))[,4] < 0.05,3:4]))
colnames(words) <- c("pvalue", "zvalue")
words

##------------------------------------------------------
accuracylr <- (imdblrCM[1]+imdblrCM[4])/sum(imdblrCM)
round(accuracylr * 100, 2)

##------------------------------------------------------
recall <- imdblrCM[4] / ( imdblrCM[4] + imdblrCM[2])
precision <- imdblrCM[4] / (imdblrCM[4] + imdblrCM[3])
F1Score_glm <- ((2* recall *precision) / (recall +precision))
F1Score_glm

##------------------------------------------------------
round(c("Accuracy_CART" = accuracyCART, "Accuracy_RF" = accuracyRF, "Accuracy_GLM" = accuracylr) * 100, 2)

round(c("F1score_CART" = F1Score_CART, "F1score_RF" = F1Score_RF, "F1score_GLM" = F1Score_glm) * 100, 2)


###########################################
######### Test Data Inclusion #############
###########################################
### CART Model ###
imdb_test <- read.csv("testdata.csv", stringsAsFactors = FALSE)
imdbtest_CART <- predict(imdbCART, data=imdb_test)
imdbtest_CART$sentiment <- imdbtest_CART[,2]  > 0.5

write.csv(imdbtest_CART$sentiment, file = "C:/Users/sarveshwaran/Desktop/New folder/all/cartmodel.csv")

### Random Forest ###
imdbtest_RF <- predict(imdbRF, data=imdb_test)
imdbtest_RF$sentiment <- imdbtest_RF>0.5

write.csv(imdbtest_RF$sentiment, file = "C:/Users/sarveshwaran/Desktop/New folder/all/RFmodel.csv")

### Generalized Linear Model ###
imdbtest_lr <-predict(imdblr, data=imdb_test)
imdbtest_lr$sentiment <- imdbtest_lr > 0.5
  
write.csv(imdbtest_lr$sentiment, file = "C:/Users/sarveshwaran/Desktop/New folder/all/lrmodel.csv")
####---------------------------------------------