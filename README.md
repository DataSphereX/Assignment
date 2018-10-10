# Assignment
Datamining on imdb scores

Reading Data:
tweets <- read.csv("training_data.csv", stringsAsFactors = FALSE)

##------------------------------------------------------------------------------------
Column Names of the data
> names(imdb)
[1] "sentiment" "review"

##------------------------------------------------------------------------------------
Tabulating the target variable

table(imdb$sentiment)

    0     1 
12500 12500 

##------------------------------------------------------------------------------------
Loding the required libraries
library(tm)
library(SnowballC)
library(wordcloud)
library(rpart)
library(rpart.plot)
library(randomForest)

##------------------------------------------------------------------------------------
Creating the corpus
corpus <- Corpus(VectorSource(imdb$review))

<Corpus is analogy to basket where we dump all the data and put them in an order, Corpus comes from tm package>

##------------------------------------------------------------------------------------
Word cloud before any preprocessing:
wordcloud(corpus, colors=rainbow(7), max.words=50)

The top words are and, the, that and for which does not help in a model understanding the true meaning of the textual content. So we do data cleaning or some pre processing

##------------------------------------------------------------------------------------
Data cleaning:
corpus <- tm_map(corpus, tolower)

#Convert to lower-case: to reduce the error by capital letter and small letter
corpus <- tm_map(corpus, removePunctuation)

# To remove the Punctuation
stopwords("english")[1:10]
[1] "i"         "me"        "my"        "myself"    "we"        "our"       "ours"     
[8] "ourselves" "you"       "your"     

corpus <- tm_map(corpus, removeWords, c("and","the", stopwords("english")))
## tn_map is a function from corpus to  remove words

corpus <- tm_map(corpus, stemDocument)

#To Stem Document: To compress a word from various tenses to a single basic word

Word cloud after preprocessing:

wordcloud(corpus, colors=rainbow(7), max.words=50)
Now the top words are “film, movi, like, time”

##------------------------------------------------------------------------------------
Creating the document term matrix:
frequencies <- DocumentTermMatrix(corpus)

##20 Low frequency words
findFreqTerms(frequencies, lowfreq = 20)

##------------------------------------------------------------------------------------
Inspect the Matrix:
inspect(frequencies[1000:1005, 505:515])
<<DocumentTermMatrix (documents: 6, terms: 11)>>
Non-/sparse entries: 2/64
Sparsity           : 97%
Maximal term length: 8
Weighting          : term frequency (tf)
Sample             :
      Terms
Docs   80s trumpet turf unfold uninspir unmitig voic wagner way without
  1000   0       0    0      0        0       0    0      0   1       0

Handling sparse matrix removing words that are having less repetition:
sparse = removeSparseTerms(frequencies, 0.920)

# After Reducing:
#Sparse nrow: 25000
#Sparse ncol: 200

##------------------------------------------------------------------------------------
imdbsparse = as.data.frame(as.matrix(sparse))
Make all variables R friendly: <By adding “X_ “ in front of all numbers and spl characters>

colnames(imdbsparse) = make.names(colnames(imdbsparse))

##------------------------------------------------------------------------------------
Adding the Dependent variable to the dataset:
imdbsparse$sentiment <- imdb$sentiment

##------------------------------------------------------------------------------------
Building CART model on sparse matrix :
imdbCART = rpart(sentiment~., data=imdbsparse, method="class")

Plot of CART:
rpart.plot(imdbCART)
 
Inferring CART Model:
Words used in review such as “Bad”, “worst”, “great”, “poor”, “bore”, “best”, “noth” has a major influence in the sentiment.

Confusion matrix:
> imdbCARTpred <- predict(imdbCART, data=imdbsparse)
> imdbCARTCM <- table("Actual"= imdb$sentiment, "prediction"=imdbCARTpred[,2] > 0.5)
> imdbCARTCM
      prediction
Actual FALSE  TRUE
     0  7459  5041
     1  2478 10022

> accuracyCART <- (imdbCARTCM[1]+imdbCARTCM[4])/sum(imdbCARTCM)
> round(accuracyCART * 100, 2)
[1] 69.92

The model accuracy of CART is 70%
F1 Score calculation: 
F1 Score = 2* (precision * recall) / (precision + recall)
F1Score
[1] 0.7272068

##------------------------------------------------------------------------------------
Building Random Forest with 20 tree limit:

imdbRF <- randomForest(sentiment~., data=imdbsparse,ntree=20)
varImpPlot(imdbRF)

Confusion Matrix
imdbRFCM
      prediction
Actual FALSE TRUE
     0  9087 3413
     1  3708 8792

> accuracyRF <- (imdbRFCM[1]+imdbRFCM[4])/sum(imdbRFCM)
> round(accuracyRF * 100, 2)
[1] 71.52

Visualizing random forest:
varImpPlot(imdbRF)
 
Inference:
Top 5 words that help classify are: 
1.	“bad”,
2.	 “worst”
3.	”great”
4.	 “poor”
5.	 “bore”

##------------------------------------------------------------------------------------
Building Logistic regression model:
Before building logistic regression we need to reduce the number of columns otherwise we will end up with p>n problem which will need refined logistic regression methods.
We will remove the words that are having less density:

sparse = removeSparseTerms(frequencies, 0.920)
imdbsparse = as.data.frame(as.matrix(sparse))
colnames(imdbsparse) = make.names(colnames(imdbsparse))
> length(names(imdbsparse))
[1] 201

The column count is now reduced.

##------------------------------------------------------------------------------------
Adding the target variable to the dataset

imdbsparse$sentiment <- imdb$sentiment

Logistic regression model:

imdblr <- glm(sentiment~., data = imdbsparse, family = "binomial")

imdblrCM <- table("Actual" = imdbsparse$sentiment, "Prediction" = imdblr$fitted.values > 0.5)

summary(imdblr)

Confusion Matrix:
> imdblrCM
      Prediction
Actual FALSE  TRUE
     0  9668  2832
1	2366 10134

McFaden R2 to evaluate the model:
> round((1 - (imdblr$deviance/imdblr$null.deviance))*100, 2)
[1] 34.69

The model explains about 34.69% of variance in the data from the number of data points we have, which is a very good score.
Summary of the model:
> summary(imdblr)

Call:
glm(formula = sentiment ~ ., family = "binomial", data = imdbsparse)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-4.1849  -0.7339   0.0000   0.7521   4.7547  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept) -0.0287809  0.0327204  -0.880 0.379075    
actual      -0.0386165  0.0365135  -1.058 0.290240    
also         0.2529213  0.0278835   9.071  < 2e-16 ***
anoth       -0.1185980  0.0409010  -2.900 0.003736 ** 
away        -0.1208985  0.0519406  -2.328 0.019932 *  
bad         -0.7556468  0.0312718 -24.164  < 2e-16 ***
[ reached getOption("max.print") -- omitted 1 row ]
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 34657  on 24999  degrees of freedom
Residual deviance: 22635  on 24799  degrees of freedom
AIC: 23037

Number of Fisher Scoring iterations: 6

> colnames(words) <- c("pvalue", "zvalue")
> words
              pvalue        zvalue
also        9.070629  1.183319e-19
anoth      -2.899638  3.735941e-03
away       -2.327627  1.993191e-02
bad       -24.163871 5.337700e-129
bit         6.344694  2.228682e-10
bore      -15.707221  1.349663e-55
call       -2.988660  2.802040e-03
cours       2.140186  3.233973e-02
differ      5.722361  1.050537e-08
director   -4.691603  2.710727e-06
dont       -5.358421  8.395258e-08
ever        3.971698  7.136203e-05
feel        4.183491  2.870664e-05
film        2.446408  1.442877e-02
final       2.356675  1.843936e-02
find        3.759276  1.704059e-04
hope       -3.368366  7.561519e-04
just      -10.087490  6.275389e-24
kid        -3.114128  1.844896e-03
know        3.090460  1.998470e-03
let        -3.054409  2.255041e-03
line       -2.989679  2.792711e-03
lot         3.161289  1.570726e-03
made       -2.553148  1.067541e-02
may         5.573670  2.494280e-08
minut     -10.278947  8.768073e-25
movi       -3.939189  8.175761e-05
one         2.550738  1.074952e-02
appear     -3.689001  2.251363e-04
better     -6.883815  5.827053e-12
good        6.814654  9.449114e-12
high        4.767285  1.867252e-06
man         3.517637  4.354080e-04
mani        2.874543  4.046128e-03
much       -3.238587  1.201234e-03
quit        3.081538  2.059340e-03
real        2.075692  3.792245e-02
run        -3.273698  1.061501e-03
scene      -2.248789  2.452595e-02
there      -2.102904  3.547419e-02
time        2.959920  3.077186e-03
action      5.052751  4.354920e-07
that       -2.476057  1.328425e-02
worth       3.451334  5.578228e-04
person      2.909484  3.620258e-03
beauti     11.495312  1.392763e-30
kill       -2.852707  4.334855e-03
live        2.795045  5.189243e-03
yet         2.358708  1.833868e-02
help        2.196027  2.809000e-02
might      -4.841595  1.288012e-06
[ reached getOption("max.print") --]

From comparing from the other 2 models there are few more words playing an important role:
“also”, “bit”, “ever”, “find”, “hope”, “just” 
has made an impact in the sentiment.

Accuracy of the model:

> accuracylr <- (imdblrCM[1]+imdblrCM[4])/sum(imdblrCM)
> round(accuracylr * 100, 2)
[1] 79.21

The Accuracy is 79.21% which is quite better than the CART and RF model.

##------------------------------------------------------------------------------------
Comparing the Accuracy of models:

> round(c("CART" = accuracyCART, "RF" = accuracyRF, "GLM" = accuracylr) * 100, 2)
 CART    RF   GLM 
69.92 71.52 79.21

F1 Scores of the Models:
> round(c("F1score_CART" = F1Score_CART, "F1score_RF" = F1Score_RF, "F1score_GLM" = F1Score_glm) * 100, 2)
F1score_CART   F1score_RF  F1score_GLM 
       72.72        71.18        79.59 


##------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------
Finding Sentiment in the Test data using CART, RF, GLM:

CART Model:


imdb_test <- read.csv("testdata.csv", stringsAsFactors = FALSE)

imdbtest_CART <- predict(imdbCART, data=imdb_test)

imdbtest_CART$sentiment <- imdbtest_CART[,2]  > 0.5

write.csv(imdbtest_CART$sentiment, file = " cartmodel.csv")

##------------------------------------------------------------------------------------
Random Forest Model:

 imdbtest_RF <- predict(imdbRF, data=imdb_test)

imdbtest_RF$sentiment <- imdbtest_RF>0.5

write.csv(imdbtest_RF$sentiment, file = "RFmodel.csv")

##------------------------------------------------------------------------------------
Generalized Linear Model:

Generalized Linear Model for test_data in .csv:
 
imdbtest_lr$sentiment <- imdbtest_lr > 0.5
  
write.csv(imdbtest_lr$sentiment, file = "C:/Users/sarveshwaran/Desktop/New folder/all/lrmodel.csv")

##------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------



















