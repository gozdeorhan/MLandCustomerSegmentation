#' ---
#' title: "HangiKredi Case Study"
#' author: "Gozde Orhan"
#' date: "20/07/2020"
#' ---

#' ### Workspace and Directory

#Remove workspace
rm(list=ls())

#Set up the working directory
setwd('/Users/gozdeorhan/Desktop/GozdeOrhan_HangiKredi')

#' ### Libraries

library(caret)
library(DMwR)
library(MLmetrics)
library(ggplot2)
library(Boruta)
library(klaR)
library(gridExtra)
library(RColorBrewer)

#' ### Dataset

#Load data
#40,000 observations, 14 variables (1 output)
df  <-  read.csv(file = "term-deposit-marketing-2020.csv", header=TRUE, na = "NA")
head(df)

#NA Values
colSums(is.na(df)) #no NA values, no imputation needed

#Class balance
table(df$y) #classes are imbalanced, balancing is a must!

#Descriptive statistics of dataset
data_sum <- summary(df)
print(data_sum)

#' # Will the customer subscribe to a term deposit?

#' ### Removing columns with zero variance and near zero variance

#Check near zero and zero variance
predictorInfo <- nearZeroVar(df,saveMetrics = TRUE)

#Column names that have zero variance
rownames(predictorInfo)[predictorInfo$zeroVar] #No predictor is removed

#Column names that have near zero variance
rownames(predictorInfo)[predictorInfo$nzv] #'Default' predictor has near zero variance, poor predictive power!

#Remove nzv column from dataset
#40,000 observations, 13 variables (1 output)
rdf <- df[,!predictorInfo$nzv]

#' ### Correlation

#Find numeric dataset
numeric_data = rdf[sapply(rdf, is.numeric)]

#Calculate correlation on numeric data
correlation_data <- cor(numeric_data,use='pairwise.complete.obs')

#Find the subset of correlation matrix with specific threshold value
#Column names that are above than certain correlation
corr_colnames <- findCorrelation(correlation_data,cutoff = 0.75,names = TRUE)
print(corr_colnames) #No significant correlation found, no predictor is removed

#' ### Dummification
#Creating dummy variables for factor variables
#' Even though most algorithms can handle categorical data, algorithms tend to learn better with dummified data 
dummies <- dummyVars(~., data = rdf[,1:12],levelsOnly = FALSE)
data_dummy <- predict(dummies, newdata = rdf)
data_dummy <- as.data.frame(data_dummy)
data_dummy <- data.frame(rdf$y, data_dummy)
names(data_dummy)[1] <- "y"
rdf <- data_dummy

#' ### Split dataset

#Randomly shuffle the data
#Set seed for reproducibility
set.seed(2408)
rdf <- rdf[sample(nrow(rdf)),]

#Splitting data as training and testing (75% training - 25% test)
split  <-  createDataPartition(rdf$y, p = 0.75, list = FALSE) #function preserves the overall class distribution of the data by default
data.train  <-  rdf[split,]
data.test  <-  rdf[-split,]

#' ### SMOTE Balancing (Training set)

#' The imbalance between classes will cause algorithms to favor majority class ('no' for this case)
#' We want to make sure that the classes are as balanced as possible, while trying not to lose too much data as well!
data.train.balanced = SMOTE(y~., data.train, perc.over = 300, k = 3, perc.under = 300)  #balance trainset
table(data.train$y)                           #check previous balance of target column
table(data.train.balanced$y)                  #check final balance of target column
dim(data.train.balanced)

#' ### Feature Selection

#For feature selection, the Boruta algorithm is utilised
boruta.output <- Boruta(y~., data = data.train.balanced, doTrace = 0)
print(boruta.output) #Since the removal of these predictors won't make much difference, they are kept

#' # ML Algorithms

#Declare control parameter to be used in training
#5-fold cross validation

fiveStats <- function(...) c(multiClassSummary(...),
                             defaultSummary(...))

ctrl <- trainControl(method = "cv",
                     number = 5,
                     classProbs = TRUE,
                     summaryFunction = fiveStats,
                     verboseIter = FALSE)

#' logLoss is used as metric during the training phase since it is important to penalise false predictions!
#' We wouldn't want to miss customers who will make a deposit!

#' ### Decision Tree

#Fit Decision Tree model
dtree <- train(y ~., data = data.train.balanced, 
               method = "rpart",
               metric ='logLoss',
               preProcess=c("center", "scale","BoxCox"), #feature values are centered, scaled and transformed in order to normalize data
               trControl=ctrl,
               tuneLength = 10)

#Evaluation
#Make class prediction on test set
dtree_prediction <- predict(dtree, newdata = data.test)

#Display confusion matrix
confusionMatrix(dtree_prediction, data.test$y, positive = "yes")

#See F1 score for the prediction above
train_f1_dtree <- F1_Score(dtree_prediction, data.test$y, positive = "yes")
print(train_f1_dtree, digits=3)

#' Let's see which feature is more important!
varImp(dtree)

#' ### Random Forest

#' mtry: Number of variables available for splitting at each tree node
#' Let algorithm find the best mtry (from 5 to 20) by itself by defining a tunegrid
tunegrid = expand.grid(.mtry= seq(from = 5, to = 10, by = 1))

#Fit Random Forest model
rf_fit <- train(y~., data = data.train.balanced, 
                method ='rf',  
                metric ='logLoss', 
                tuneGrid = tunegrid, 
                trControl = ctrl,
                preProcess = c("center", "scale","BoxCox")) #feature values are centered, scaled and transformed in order to normalize data

#Evaluation
#Make class prediction on test set
rf_prediction <- predict(rf_fit, newdata = data.test)

#Display confusion matrix
confusionMatrix(rf_prediction, data.test$y, positive = "yes")

#See F1 score for the prediction above
train_f1_rf <- F1_Score(rf_prediction, data.test$y, positive = "yes")
print(train_f1_rf, digits=3)

#' Let's see which feature is more important!
varImp(rf_fit)

#' For predicting the desired outcome, it is seen that age, existing loans, education and marital status are important features with significant predictive powers.

#' ### NNet

#' size: The number of units in hidden layer
#' decay: Regularization parameter to avoid over-fitting
#' Let algorithm find the best size and decay by itself by defining a tunegrid
grid = expand.grid(size = seq(from = 1, to = 10, by = 1),
                   decay = seq(from = 0.1, to = 0.5, by = 0.1))

#Fit single-layer Neural Network model
nnet_fit = train(y~., data = data.train.balanced, 
                 method ='nnet',  
                 metric ='logLoss', 
                 tuneGrid = grid, 
                 trControl = ctrl,
                 preProcess = c("center", "scale","BoxCox")) #feature values are centered, scaled and transformed in order to normalize data

#Evaluation
#Make class prediction on test set
nnet_prediction <- predict(nnet_fit, newdata = data.test)

#Display confusion matrix
confusionMatrix(nnet_prediction, data.test$y, positive = "yes")

#See F1 score for the prediction above
train_f1_nnet <- F1_Score(nnet_prediction, data.test$y, positive = "yes")
print(train_f1_nnet, digits=3)

#' # Customer Segmentation
#' ### Dataset

age_break <- c(-Inf, 30, 50, 65, Inf)
age_group <- c("YoungAdult", "Adult", "MiddleAged", "Elderly")
df$age.cat <- cut(df$age, breaks = age_break, labels = age_group)

balance_break <- c(-Inf, 200, 2000, Inf)
balance_group <- c("Low", "Medium", "High")
df$balance.cat <- cut(df$balance, breaks = balance_break, labels = balance_group)

duration_break <- c(-Inf, 120, 400, Inf)
duration_group <- c("Short", "Average", "Long")
df$duration.cat <- cut(df$duration, breaks = duration_break, labels = duration_group)

df <- df[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,14)]
head(df)

categorical <- df[sapply(df, is.factor)]

#' ### K-Modes Clustering

cat_clusters <- kmodes(categorical, modes = 3, iter.max = 5000)

#Create a column to represent cluster numbers
categorical$cluster <- as.character(cat_clusters$cluster)
print(cat_clusters$size)
head(categorical)

#' ### Which group should we focus on?
#' Let's see!

#' Randomly select three predictors are: education, marital and contact

#A plot to see which clusters made the term deposit 
main <- ggplot(categorical, aes(x=y, fill=cluster)) +
  geom_bar(stat="count", position=position_dodge())+theme_minimal()

#Education vs term deposit
plot_education1 <- ggplot(categorical, aes(x=y, fill=education)) +
  geom_bar(stat="count", position=position_dodge())+theme_minimal()
plot_education1 <- plot_education1+scale_fill_brewer(palette="Spectral")

plot_education2 <- ggplot(categorical, aes(x=education, fill=cluster)) +
  geom_bar(stat="count", position=position_dodge())+theme_minimal()

grid.arrange(main, plot_education1, plot_education2)

#' Note that first plot shows that the people in cluster 1 and 2 made the term deposit, and the second plot shows that the people with secondary and tertiary education made the term deposit.
#' Third plot shows that people with respective education level are in cluster 1 and 2. Hence we can say that people in 1 and 2 may be more inclined to buy new products. The logic is same for the following graphs.

#Marital vs term deposit
plot_marital1 <- ggplot(categorical, aes(x=y, fill=marital)) +
  geom_bar(stat="count", position=position_dodge())+theme_minimal()
plot_marital1 <- plot_marital1+scale_fill_brewer(palette="Spectral")

plot_marital2 <- ggplot(categorical, aes(x=marital, fill=cluster)) +
  geom_bar(stat="count", position=position_dodge())+theme_minimal()

grid.arrange(main, plot_marital1, plot_marital2)

#Contact vs term deposit
plot_contact1 <- ggplot(categorical, aes(x=y, fill=contact)) +
  geom_bar(stat="count", position=position_dodge())+theme_minimal()
plot_contact1 <- plot_contact1+scale_fill_brewer(palette="Spectral")

plot_contact2 <- ggplot(categorical, aes(x=contact, fill=cluster)) +
  geom_bar(stat="count", position=position_dodge())+theme_minimal()

grid.arrange(main, plot_contact1, plot_contact2)

#' The observation can be made from these plots are that the ones who made the term deposit commitment is mostly in cluster number 1 and 2, thus they can be more inclined to buy new products!
#' Focusing on group 1 and 2 may be more profitable for clients!