#The Insurance Company Benchmark (COIL 2000)
rm(list=ls()) #will remove ALL objects 
library(data.table)
C.train <- fread('http://kdd.ics.uci.edu/databases/tic/ticdata2000.txt')
head(C.train)
str(C.train)
C.test <- fread('http://kdd.ics.uci.edu/databases/tic/ticeval2000.txt')
head(C.test)
str(C.test)
table(C.test$V85)
C.target_test <- fread('http://kdd.ics.uci.edu/databases/tic/tictgts2000.txt')
str(C.target_test)
summary(C.target_test)
table(C.target_test)


ytrain = as.factor(C.train$V86) ## randomForest() requires categorical outcome to be a factor
ytest = as.factor(C.target_test$V1)
Xtrain = C.train[,1:85]
Xtest = C.test
str(Xtrain)
str(Xtest)

table(ytrain)

apply(is.na(C.train), 2, sum) ## check which variable has missing data
apply(is.na(C.train), 1, sum) ## check which observation has missing data

# Random Forests and Bagging
library(randomForest)

set.seed(1)
rf1 = randomForest(ytrain ~ ., data=Xtrain, ntree=1000)
plot(rf1)
plot(rf1$err.rate[,1], type='l', xlab='trees', ylab='Error')
rf1

names(rf1) ## all details are here
rf1$mtry; rf1$ntree ## check what default values were used
rf1$confusion ## same as table(Heart2.train$AHD, rf$predicted)
rf1$err.rate[rf1$ntree, ] ## the FPR and FNR can also be obtained here

importance(rf1) ## show rf$importance
varImpPlot(rf1) ## same as dotchart(rf$importance[, 'MeanDecreaseGini']) except order


varImpPlot(randomForest(ytrain ~ ., data=Xtrain)) ## repeat a few times

set.seed(1)
varImpPlot(randomForest(ytrain ~ ., data=Xtrain, mtry=1))
varImpPlot(randomForest(ytrain ~ ., data=Xtrain, mtry=5))
varImpPlot(randomForest(ytrain ~ ., data=Xtrain, mtry=15))

rf2 = randomForest(ytrain ~ ., data=Xtrain,mtry=5, importance=T)
rf2$importance
importance(rf2) ## different from rf$importance except the last column
varImpPlot(rf2)
rf2
plot(rf2)
rf2$mtry; rf2$ntree ## check what default values were used
rf2$confusion ## same as table 
rf2$err.rate[rf2$ntree, ]  


set.seed(1)
rf3 = randomForest(ytrain ~ ., data=Xtrain,mtry=5, importance=T,ntree=500)
rf3
plot(rf3)
varImpPlot(rf3)


#Cross-validation to select m
library(caret)
cvCtrl = trainControl(method="repeatedcv", number=5, repeats=4, ## 5-fold CV repeated 4 times
                      #summaryFunction=twoClassSummary,
                      classProbs=FALSE)
set.seed(1)
fitRFcaret = train(x=Xtrain[, 1:85], y=ytrain, trControl=cvCtrl,
                   tuneGrid=data.frame(mtry=1:15),
                   #tuneLength=4,
                   #metric="ROC", ## when summaryFunction=twoClassSummary
                   method="rf", ntree=500) ##  The final value used for the model was mtry = 1.
fitRFcaret
plot(fitRFcaret)

names(fitRFcaret)
fitRFcaret$results
fitRFcaret$besXtestune$mtry
fitRFcaret$finalModel
fitRFcaret$finalModel$confusion ## OOB confusion matrix

###Best model
set.seed(1)
rff = randomForest(ytest ~ ., data=Xtest,mtry=1, importance=T,ntree=500)
rff





#Boosting
library(survival)
library(splines)
library(parallel)
library(gbm)
bt1 = gbm(ytrain ~ ., data=Xtrain, distribution="gaussian", n.trees=500)
bt1
names(bt1)
bt1$interaction.depth ## stumps
bt1$cv.folds ## no CV was done
bt1pre=predict(bt1, Xtrain, n.trees=500)

bt4 = gbm(ytrain ~ ., data=Xtrain, distribution="gaussian", n.trees=500, interaction.depth=4)

summary(bt1) ## results and a plot
summary(bt4) ## with d=4, the influence 
summary(bt1, plotit=F) ## without the plot
mse = function(a,b) mean((a-b)^2)
ytrain <- as.numeric(ytrain)
mse(ytrain, predict(bt1, Xtrain, n.trees=500)) ## MSE= 
mse(ytrain, predict(bt4, Xtrain, n.trees=500)) ## MSE= 

sum(summary(bt1, plotit=F)$rel.inf) ## 100
sum(summary(bt4, plotit=F)$rel.inf) ## 100

bt.try = gbm(ytrain ~ ., data=Xtrain, distribution="gaussian", n.trees=100, bag.fraction=1)
summary(bt.try, plotit=F) 


#levels(ytest)
#levels(ytrain)[1] <- "0" 
#levels(ytrain)[2] <- "1"
#levels(ytrain)
### look at model performance at the end of 1000, 2000, etc. trees.
set.seed(1)
bt4b = gbm(ytrain ~ ., data=Xtrain, distribution="gaussian", n.trees=500, interaction.depth=4,shrinkage =0.01)
mse(ytrain, predict(bt4b, Xtrain, n.trees=500))
bt4c = gbm(ytrain ~ ., data=Xtrain, distribution="gaussian", n.trees=1000, interaction.depth=4,shrinkage =0.01)
mse(ytrain, predict(bt4c, Xtrain, n.trees=1000))
bt4d = gbm(ytrain ~ ., data=Xtrain, distribution="gaussian", n.trees=1000, interaction.depth=4,shrinkage =0.1)
mse(ytrain, predict(bt4d, Xtrain, n.trees=1000))
bt4e = gbm(ytrain ~ ., data=Xtrain, distribution="gaussian", n.trees=1000, interaction.depth=8,shrinkage =0.1)
mse(ytrain, predict(bt4e, Xtrain, n.trees=1000))
#Cross-validation using caret
ytrain = as.factor(C.train$V86)
library(caret)
set.seed(1)
ctr = trainControl(method="cv", number=5) ## 5-fold CV
mygrid = expand.grid(n.trees=seq(500, 1000, 100), interaction.depth=1:10,
                     shrinkage=0.1, n.minobsinnode=5)
boost.caretk <- train(x=Xtrain, y=ytrain, trControl=ctr, method='gbm',
                      tuneGrid=mygrid,
                      preProc=c('center','scale'), verbose=F)
boost.caretk
plot(boost.caretk)

#Using the optimal hyperparameters selected by train() improves the result!
boost.caretk$besXtestune
names(boost.caretk)
boost.caretk$results
boost.caretk$finalModel

###Best model
btf = gbm(ytest ~ ., data=Xtest, distribution="gaussian", n.trees=800, interaction.depth=1,shrinkage =0.1)
btf
names(btf)
test.pred <- predict(btf, Xtest, n.trees=800)
ytest <- as.numeric(ytest)
mse(ytest, test.pred)
ytest = as.factor(C.target_test$V1)

###SVM
library(e1071)
svmfit1 = svm(ytrain ~ ., data=Xtrain, kernel='linear')
svmfit1



# support vector machine with a radial kernel. Use the default value for gamma.
sapply(Xtrain, function(x) sum(is.na(x)))
sapply(ytrain, function(x) sum(is.na(x)))
colSums(is.na(Xtrain))
set.seed(1)
tune.out <- tune(svm, ytrain ~ ., data=cbind(Xtrain,ytrain), kernel = "radial" ,ranges=list(cost=2^(-5:5), gamma=2^(-5:0)))
tune.out$best.model
summary(tune.out)

svm.radial <- svm(ytrain ~ ., data=cbind(Xtrain,ytrain), kernel = "radial",  cost = tune.out$best.parameter$cost)
summary(svm.radial)
train.pred <- predict(svm.radial, Xtrain)
table(ytrain, train.pred)
test.pred <- predict(svm.radial, Xtest)
table(ytest, test.pred)

# ( using a support vector machine with a polynomial kernel. Set degree=2.
svm.poly0 <- svm(ytrain ~ ., data=cbind(Xtrain,ytrain), kernel = "polynomial", degree = 2)
summary(svm.poly0)
train.pred <- predict(svm.poly0, Xtrain)
table(ytrain, train.pred)

set.seed(1)
tune.outp <- tune(svm, ytrain ~ ., data=cbind(Xtrain,ytrain), kernel = "polynomial", degree = 2, ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.outp)

set.seed(1)
svm.poly <- svm(ytrain ~ ., data=cbind(Xtrain,ytrain), kernel = "polynomial", degree = 2, cost = tune.outp$best.parameter$cost)
summary(svm.poly)
train.pred <- predict(svm.poly, Xtrain)
table(ytrain, train.pred)
test.pred <- predict(svm.poly, Xtest)
table(ytest, test.pred)



#####2.1	K-means
aa = kmeans(Xtrain, 2, nstart=10); aa$tot.withinss ## repeat to see the effect of nstart
all.equal(aa$tot.withinss, sum((Xtrain - aa$centers[aa$cluster,])^2)) ## True
all.equal(aa$totss, sum((t(Xtrain) - apply(Xtrain,2,mean))^2)) ## True
## standardize the features
Xtrain2 = scale(Xtrain)
bb = kmeans(Xtrain2, 2, nstart=10)
str(bb)
table(aa$cluster, bb$cluster) 

dist1 = dist(Xtrain2)
str(dist1) ##  
sqrt(sum((Xtrain2[1,]-Xtrain2[2,])^2)) ## dist between 1st and 2nd observations
cluster1 = cutree(hclust(dist1), 5)
cluster2 = cutree(hclust(dist1), h=5.7)
table(cluster1); table(cluster2)
table(cluster1, cluster2) ## same clustering
plot(hclust(dist1, method='complete'), labels=F, xlab='', main="Complete")
plot(hclust(dist1, method='average'), labels=F, xlab='', main="Average")
plot(hclust(dist1, method='single'), labels=F, xlab='', main="Single")

#try the correlation approach.
dist2 = as.dist(1 - cor(t(Xtrain2))) ## dist = 1 - Pearson correlation
plot(hclust(dist2, method='complete'), labels=F, xlab='', main="Complete")
plot(hclust(dist2, method='average'), labels=F, xlab='', main="Average")
plot(hclust(dist2, method='single'), labels=F, xlab='', main="Single")



####
sd.data = scale(Xtrain) ## standardize the columns
data.dist = dist(sd.data) ## default is Euclidean distance
attributes(data.dist)
## Hierarchical clustering using single linkage
hc2 = hclust(data.dist, method="single")
plot(hc2, labels=F, main="Single Linkage", xlab="", ylab="", sub="")
## Hierarchical clustering using average linkage
hc3 = hclust(data.dist, method="average")
plot(hc3, labels=F, main="Average Linkage", xlab="", ylab="", sub="")

#Use only the variables 6-41
Xtrain3=C.train[,6:41]
Xtrain4=scale(Xtrain3)
cc = kmeans(Xtrain4, 2, nstart=10)

dist3 = dist(Xtrain4)
str(dist3) ##  
sqrt(sum((Xtrain4[1,]-Xtrain4[2,])^2)) ## dist between 1st and 2nd observations
cluster1 = cutree(hclust(dist1), 5)
cluster2 = cutree(hclust(dist1), h=5.7)
table(cluster1); table(cluster2)
table(cluster1, cluster2) ## same clustering
plot(hclust(dist3, method='complete'), labels=F, xlab='', main="Complete")
plot(hclust(dist3, method='average'), labels=F, xlab='', main="Average")
plot(hclust(dist3, method='single'), labels=F, xlab='', main="Single")

sd.data2 = scale(Xtrain3) ## standardize the columns
data.dist2 = dist(sd.data2) ## default is Euclidean distance
attributes(data.dist2)
## Hierarchical clustering using single linkage
hc22 = hclust(data.dist2, method="single")
plot(hc22, labels=F, main="Single Linkage", xlab="", ylab="", sub="")
## Hierarchical clustering using average linkage
hc23 = hclust(data.dist2, method="average")
plot(hc23, labels=F, main="Average Linkage", xlab="", ylab="", sub="")
## Hierarchical clustering using average linkage
hc24 = hclust(data.dist2, method="complete")
plot(hc24, labels=F, main="complete", xlab="", ylab="", sub="")