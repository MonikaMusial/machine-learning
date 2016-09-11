library(caret)
library(AppliedPredictiveModeling)

#Reading data

training<-read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"), stringsAsFactors = FALSE)
finalTest<-read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"), stringsAsFactors = FALSE)

tsnr<-nrow(finalTest)
trnr<-nrow(training)
finalTest$classe<-1
training$problem_id<-1

# I am going to bind two datasets to remove Nas and remain only numeric columns
trainingSet<-rbind(training,finalTest)
#removing user name
trainingSet<-trainingSet[,-1]

# I am removing columns that are all Nas
trainingSet<-subset(trainingSet, select = colSums(!is.na(trainingSet))>0)

# In this study I am going to concentrate only on numerical variables

training2<-trainingSet[,-112]
nonNumeric=c()
for (name in colnames(training2))
{
  if(is.factor(training2[,name]))     training2[,name]<-as.numeric(as.character(training2[,name]))
  nonNumeric <- c(nonNumeric,is.numeric(training2[,name]))
}
training2<-training2[,nonNumeric]

# I am removing columns that are all Nas
training2<-subset(training2, select = colSums(!is.na(training2))>0)

#impute missing values with median for test and training set
for (name in names(training2)) {
  training2[,name][is.na(training2[,name])] <-median(as.numeric(training2[,name]), na.rm = TRUE)
  training2[,name] <- ifelse(training2[,name] == 0, median(training2[,name], na.rm = TRUE),                                   training2[,name]) 
}

metric <- "Accuracy"
#In this case study I will use 10-fold cross validation with 3 repeats
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# I assign a random number seed to a variable, so that we can re-set the random number generator before I train each algorithm
seed <- 1721

# adding some noise to data to avoid constant values in columns
training2 <- training2 + rnorm(17*3)

#seperating 20 test cases
finalTest<-training2[-(1:trnr),]
training2 <- training2[1:trnr,]

#divide the training data into test and training
inTrain = createDataPartition(training$classe, p = 3/4)[[1]]
pca.train = training2[ inTrain,]
pca.test = training2[-inTrain,]

#principal component analysis
pca.train <- data.frame(t(na.omit(t(pca.train))))
prin_comp <- prcomp(pca.train, scale. = T)
#compute standard deviation of each principal component
std_dev <- prin_comp$sdev
#compute variance
pr_var <- std_dev^2
#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
#cumulative plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
#This plot shows that 60 components results in variance close to ~ 98%. Therefore, in this case, I will select number of components as 60 [PC1 to PC60] and proceed to the modeling stage.

datasetTrain=data.frame(classe = trainingSet[inTrain,"classe"], prin_comp$x)
datasetTrain<-datasetTrain[,1:61]

datasetTrain$classe<-factor(datasetTrain$classe)

#The algorithms that i will be testing are:

# kNN
set.seed(seed)
fit.knn <- train(classe~., data=datasetTrain, method="knn", metric=metric,  trControl=control)

# Random Forest
set.seed(seed)
fit.rf <- train(classe~., data=datasetTrain, method="rf", metric=metric, trControl=control)
# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(classe~., data=datasetTrain, method="gbm", metric=metric, trControl=control)
# Logistic Regression
set.seed(seed)
fit.glm <- train(classe~., data=datasetTrain, method="glm", metric=metric, trControl=control)

#comparison
results <- resamples(list(logistic=fit.glm, knn=fit.knn,  
                          rf=fit.rf, gbm=fit.gbm))
# Table comparison
summary(results)

#transform test into PCA
test.data <- predict(prin_comp, newdata = pca.test)
test.data <- as.data.frame(test.data)
#select the first 60 components
test.data <- test.data[,1:60]

#make prediction on my test data
prediction <- predict(fit.knn, test.data)
cm<-confusionMatrix(prediction, training[-inTrain,"classe"])
cm$overall
#out of sample error
outOfSampleError.accuracy <- sum(prediction == training[-inTrain,"classe"])/length(prediction)
outOfSampleError.accuracy 

# I will use my prediction model to predict 20 test cases:

test.data <- predict(prin_comp, newdata = finalTest)
test.data <- as.data.frame(test.data)
#select the first 60 components
test.data <- test.data[,1:60]

#make prediction on my test data
prediction <- predict(fit.knn, test.data)
prediction

