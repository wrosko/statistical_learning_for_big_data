library(klaR)
library(readr)
library(factoextra)
library(cluster) 
library(fpc)
library(NbClust)
library(mclust)
library(caret)
library(rattle)
library(gridExtra)
library(xtable)
library(klaR)
library(mda)

setwd('d:/chalmers/statistical_learning_for_big_data')
data1 <- read.table("data/HTRU_2.csv", sep=",", header=F)
data2 <- read.table("data/Skin_NonSkin.csv", sep=",", header=F)

# Function ##########
run_ml_models <- function(train,train.y,test,test.y,seed,control,metric,preProcess){
  
  #glm
  set.seed(seed)
  fit.glm <- train(x=train, y =train.y, method="glm",metric=metric,trControl=control)
  summary(fit.glm)
  test_pred.glm <-predict(fit.glm, newdata = test)
  confusionMatrix(test_pred.glm, test.y)
  
  #kNN
  set.seed(seed)
  fit.knn <- train(x=train,y=train.y, method="knn", metric=metric, preProc=preProcess, trControl=control)
  summary(fit.knn)
  test_pred.knn <-predict(fit.knn, newdata = test)
  # confusionMatrix(test_pred.knn, test.y)
  
  
  # CART
  set.seed(seed)
  fit.cart <- train(x=train,y=train.y, method="rpart", metric=metric, trControl=control)
  summary(fit.cart)
  test_pred.cart <-predict(fit.cart, newdata = test)
  # confusionMatrix(test_pred.cart, test.y)
  
  # Random Forest
  set.seed(seed)
  start_time <- Sys.time()
  fit.rf <- train(x=train,y=train.y, method="rf", metric=metric, trControl=control)
  end_time <- Sys.time()
  end_time - start_time
  
  # LDA
  set.seed(seed)
  fit.lda <- train(x=train,y=train.y, method="lda", metric=metric, trControl=control)
  summary(fit.lda)
  test_pred.lda <-predict(fit.lda, newdata = test)
  # confusionMatrix(test_pred.lda, test.y)
  
  #  qda
  set.seed(seed)
  fit.qda <- train(x=train,y=train.y, method="qda", metric=metric, trControl=control)
  summary(fit.qda)
  test_pred.qda <-predict(fit.qda, newdata = test)
  # confusionMatrix(test_pred.qda, test.y)
  
  # pda
  set.seed(seed)
  fit.pda <- train(x=train,y=train.y, method="pda", metric=metric, trControl=control)
  summary(fit.pda)
  test_pred.pda <-predict(fit.pda, newdata = test)
  # confusionMatrix(test_pred.pda, test.y)
  
  # nb
  set.seed(seed)
  fit.nb <- train(x=train,y=train.y, method="nb", metric=metric, trControl=control)
  summary(fit.nb)
  test_pred.nb <-predict(fit.nb, newdata = test)
  # confusionMatrix(test_pred.nb, test.y)
  
  
  
  
  summary(fit.rf)
  test_pred.rf <-predict(fit.rf, newdata = test)
  # confusionMatrix(test_pred.rf, test.y)
  
  
  results<- resamples(list(rf=fit.rf,cart=fit.cart,knn=fit.knn,
                           logistic=fit.glm,naive_bayes=fit.nb,
                           penalized_DA = fit.pda, quadratic_DA= fit.qda,
                           linear_DA = fit.lda))
  return(results)
}


control <- trainControl(method="repeatedcv", number=2, repeats=1)
# control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 25

metric <- "Accuracy"

preProcess = c("center","scale")

# First run models on data1

smp_size <- floor(0.8 * nrow(data1))
## set the seed to make  partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data1)), size = smp_size)
train1 <- data1[train_ind, ]
test1 <- data1[-train_ind, ]

train1.y <- train1$V9
train1$V9 <- NULL
train1.y <- factor(train1.y, labels = c("no", "yes"))
test1.y <- test1$V9
test1$V9 <- NULL
test1.y <- factor(test1.y, labels = c("no", "yes"))

pulsar_results <- run_ml_models(train1,train1.y,test1,test1.y,seed,control,metric,preProcess)

# plot(fit.knn, main="Neighbors vs. Accuracy for kNN: Pulsars")

# fancyRpartPlot(fit.cart$finalModel,
#                main = "CART Mode: Pulsars")



##DATA2 - Skin###################################################################################

# Skin segmentation had many instances, I decided to use 40000 random samples
data2 <- read.table("data/Skin_NonSkin.csv", sep=",", header=F)
set.seed(seed)
num_samp = 40000
sample_inds <- sample(seq_len(nrow(data2)), size = num_samp)
data2 <- data2[sample_inds,]
# sum(data2$V4)/nrow(data2)
sample_size <- floor(0.8 * num_samp)
## set the seed to make  partition reproducible
set.seed(123)
train_ind2 <- sample(seq_len(nrow(data2)), size = sample_size)
train2 <- data2[train_ind2, ]
test2 <- data2[-train_ind2, ]

train2.y <- train2$V4
train2$V4 <- NULL
train2.y <- factor(train2.y, labels = c("no", "yes"))

test2.y <- test2$V4
test2$V4 <- NULL
test2.y <- factor(test2.y, labels = c("no", "yes"))

results2 <- run_ml_models(train2,train2.y,test2,test2.y,seed,control,metric,preProcess)



# Results #######################################################################
summary(pulsar_results)
#boxplot
bwplot(pulsar_results,main="Accuracy and Kappa results for the various methods: Pulsar")
#dot plot
dotplot(pulsar_results,main="Accuracy and Kappa results for the various methods: Pulsar")

summary(results2)
#boxplotskin
bwplot(results2,main="Accuracy and Kappa results for the various methods: Skin Segmentation")
#dot plot
dotplot(results2,main="Accuracy and Kappa results for the various methods: Skin")




