library(caret)
library(psych)
library(glmnet)#
library(pamr)#
library(sparseLDA)#
library(pamr)#
library(mlr)#
library(FSelector)#
library(MASS)
library(beepr)

### This is the worst written script in this assignment due to it being hacked together. Just a warning
##### Functions ####
standardize<-function(x) {
  x<-(x-mean(x))/sd(x)
}

perform_analyses <- function(i){
  blob_string <-  paste("data/mini3/artificial_blobs",i,"centers_v2.csv",sep="_")
  mydata <- read.table(blob_string, sep=",", header=F)
  
  # define label and num. unrelated features
  classlab<-7 # which column is the label
  unrel<-10 #### how many unrelated features
  
  mydata[,-classlab]<-apply(mydata[,-classlab],2,standardize) # standardize features
  
  # Generate unrelated features
  set.seed(10)
  newdata<-matrix(rnorm(dim(mydata)[1]*unrel),dim(mydata[1]),unrel) # unrelated features
  pv<-dim(mydata)[2]
  mydata<-cbind(mydata,newdata)
  names(mydata)[(pv+1):dim(mydata)[2]]<-paste("A",seq(1,unrel),sep=".")
  
  # Set class name
  names(mydata)[classlab]<-"Class"
  splitprop<-.25 #How much data to train on...
  # split data
  ii<-createDataPartition(mydata[,classlab],p=splitprop,list=F)
  #
  x.train<-mydata[ii,-classlab]
  x.test<-mydata[-ii,-classlab]
  y<-as.factor(mydata[,classlab])
  y.train<-y[ii]
  y.test<-y[-ii]
  #
  #### data split to try methods on
  ### RFE (recursive feature elemination through the caret package) ##############
  #
  ctrl <- rfeControl(functions = ldaFuncs,
                     method = "cv", # look for other options in the help file
                     verbose = FALSE)
  
  ldaProfile <- rfe(x.train, y.train,
                    sizes=1:dim(x.train)[2],
                    rfeControl = ctrl)
  
  plot(ldaProfile, type = c("o", "g"))
  
  ldaProfile
  
  Anames<-c("LDA") # store method name
  blob_performance<-postResample(predict(ldaProfile, x.test), y.test) # store test performance
  FS<-matrix(0,1,dim(x.train)[2])
  usewhich<-apply(matrix(names(x.train),length(names(x.train)),1),1,is.element,
                  set=ldaProfile$optVariables) # which were selected
  FS[usewhich]<-1
  FS
  #########################
  
  ctrl <- rfeControl(functions = nbFuncs,
                     method = "cv",
                     verbose = FALSE)
  
  nbProfile <- rfe(x.train, y.train,
                   sizes=1:dim(x.train)[2],
                   rfeControl = ctrl)
  
  plot(nbProfile, type = c("o", "g"))
  
  nbProfile
  Anames<-c(Anames,"NB")
  blob_performance<-rbind(blob_performance,postResample(predict(nbProfile, x.test), y.test))
  usewhich<-apply(matrix(names(x.train),length(names(x.train)),1),1,is.element,set=nbProfile$optVariables) # which were selected
  FS<-rbind(FS,rep(0,length(names(x.train))))
  FS[dim(FS)[1],usewhich]<-1
  ##############################
  
  ctrl <- rfeControl(functions = rfFuncs,
                     method = "cv",
                     verbose = FALSE)
  
  rfProfile <- rfe(x.train, y.train,
                   sizes=1:dim(x.train)[2],
                   rfeControl = ctrl)
  
  plot(rfProfile, type = c("o", "g"))
  
  rfProfile
  Anames<-c(Anames,"RF")
  blob_performance<-rbind(blob_performance,postResample(predict(rfProfile, x.test), y.test))
  usewhich<-apply(matrix(names(x.train),length(names(x.train)),1),1,is.element,set=rfProfile$optVariables) # which were selected
  FS<-rbind(FS,rep(0,length(names(x.train))))
  FS[dim(FS)[1],usewhich]<-1
  
  
  
  ####### shrunken centroids######################################
  cc<-train(x=x.train,y=y.train,method="pam")
  confusionMatrix(cc)
  plot(cc)
  # training error
  pp<-predict(cc,x.test)
  confusionMatrix(pp,y.test)
  cc$finalModel$centroids
  cc$bestTune
  tt<-cc$finalModel$centroids
  ts<-as.matrix(abs(tt))-as.numeric(cc$bestTune)
  ts[ts<0]<-0
  ts<-sign(tt)*ts
  ts<-apply(abs(ts),1,sum)
  # test error
  pp<-predict(cc,x.test,type="prob")
  pp$obs<-y.test
  head(pp)
  pp$pred<-predict(cc,x.test)
  Anames<-c(Anames,"ShrCent")
  blob_performance<-rbind(blob_performance,multiClassSummary(pp,lev=levels(pp$obs))[4:5])
  FS<-rbind(FS,rep(0,length(names(x.train))))
  FS[dim(FS)[1],ts!=0]<-1
  
  ### sparseLDA ######################################
  cc<-train(x=x.train,y=y.train,method="sparseLDA")
  confusionMatrix(cc)
  plot(cc)
  # training error
  pp<-predict(cc,x.test)
  confusionMatrix(pp,y.test)
  cc$finalModel
  # test error
  pp<-predict(cc,x.test,type="prob")
  pp$obs<-y.test
  head(pp)
  pp$pred<-predict(cc,x.test)
  Anames<-c(Anames,"spLDA")
  blob_performance<-rbind(blob_performance,multiClassSummary(pp,lev=levels(pp$obs))[4:5])
  FS<-rbind(FS,rep(0,length(names(x.train))))
  FS[dim(FS)[1],cc$finalModel$varIndex]<-1
  
  #### sparse logistic regression ######################################
  cc<-train(x=x.train,y=y.train,method="glmnet")
  confusionMatrix(cc)
  plot(cc)
  # training error
  pp<-predict(cc,x.test)
  confusionMatrix(pp,y.test)
  coef(cc$finalModel,cc$bestTune$lambda)
  if (length(unique(mydata$Class))>2) {
    BB<-Reduce("+",coef(cc$finalModel,cc$bestTune$lambda)) }
  if (length(unique(mydata$Class))==2) {
    BB<-coef(cc$finalModel,cc$bestTune$lambda) }
  #  
  FS<-rbind(FS,rep(0,length(names(x.train))))
  FS[dim(FS)[1],as.vector(BB)[-1]!=0]<-1
  # test error
  pp<-predict(cc,x.test,type="prob")
  pp$obs<-y.test
  head(pp)
  pp$pred<-predict(cc,x.test)
  Anames<-c(Anames,"spLogcaret")
  blob_performance<-rbind(blob_performance,multiClassSummary(pp,lev=levels(pp$obs))[4:5])
  ## instead of caret - using built in cv function in glmnet
  typeclass<-"multinomial"
  if (length(unique(mydata$Class))==2) {
    typeclass<-"binomial"
  }
  #
  gg<-glmnet(x=as.matrix(x.train),y=y.train,family=typeclass)
  plot(gg,xvar="lambda")
  cv.gg<-cv.glmnet(x=as.matrix(x.train),y=y.train,family=typeclass)
  plot(cv.gg)
  names(cv.gg)
  coef(cv.gg,s="lambda.1se")
  coef(cv.gg,s="lambda.min")
  #
  if (length(unique(mydata$Class))>2) {
    BB<-Reduce("+",coef(cv.gg,s="lambda.1se")) }
  if (length(unique(mydata$Class))==2) {
    BB<-coef(cv.gg,s="lambda.1se") }
  #  
  FS<-rbind(FS,rep(0,length(names(x.train))))
  FS[dim(FS)[1],as.vector(BB)[-1]!=0]<-1
  #
  pp<-predict(gg,s=cv.gg$lambda.1se,type="class",newx=as.matrix(x.test))
  acc<-length(y.test[pp==y.test])/length(y.test)
  kap<-cohen.kappa(table(pp,y.test),n.obs=length(y.test))
  blob_performance<-rbind(blob_performance,c(acc,kap$kappa))
  Anames<-c(Anames,"spLog1SE")
  #
  if (length(unique(mydata$Class))>2) {
    BB<-Reduce("+",coef(cv.gg,s="lambda.min")) }
  if (length(unique(mydata$Class))==2) {
    BB<-coef(cv.gg,s="lambda.min") }
  FS<-rbind(FS,rep(0,length(names(x.train))))
  FS[dim(FS)[1],as.vector(BB)[-1]!=0]<-1
  #
  pp<-predict(gg,s=cv.gg$lambda.min,type="class",newx=as.matrix(x.test))
  acc<-length(y.test[pp==y.test])/length(y.test)
  kap<-cohen.kappa(table(pp,y.test),n.obs=length(y.test))
  blob_performance<-rbind(blob_performance,c(acc,kap$kappa))
  Anames<-c(Anames,"spLog1min")
  # try grouped also (for more than 2 classes)
  if (length(unique(mydata$Class))>2) {
    gg<-glmnet(x=as.matrix(x.train),y=y.train,family="multinomial",type.multinomial="grouped")
    plot(gg,xvar="lambda")
    cv.gg<-cv.glmnet(x=as.matrix(x.train),y=y.train,family="multinomial",type.multinomial="grouped")
    plot(cv.gg)
    names(cv.gg)
    coef(cv.gg,s="lambda.1se")
    coef(cv.gg,s="lambda.min")
    #
    if (length(unique(mydata$Class))>2) {
      BB<-Reduce("+",coef(cv.gg,s="lambda.1se")) }
    if (length(unique(mydata$Class))==2) {
      BB<-coef(cv.gg,s="lambda.1se") }
    FS<-rbind(FS,rep(0,length(names(x.train))))
    FS[dim(FS)[1],as.vector(BB)[-1]!=0]<-1
    pp<-predict(gg,s=cv.gg$lambda.1se,type="class",newx=as.matrix(x.test))
    acc<-length(y.test[pp==y.test])/length(y.test)
    kap<-cohen.kappa(table(pp,y.test),n.obs=length(y.test))
    blob_performance<-rbind(blob_performance,c(acc,kap$kappa))
    Anames<-c(Anames,"spLogGroup1se")
    #
    if (length(unique(mydata$Class))>2) {
      BB<-Reduce("+",coef(cv.gg,s="lambda.min")) }
    if (length(unique(mydata$Class))==2) {
      BB<-coef(cv.gg,s="lambda.min") }
    FS<-rbind(FS,rep(0,length(names(x.train))))
    FS[dim(FS)[1],as.vector(BB)[-1]!=0]<-1
    pp<-predict(gg,s=cv.gg$lambda.min,type="class",newx=as.matrix(x.test))
    acc<-length(y.test[pp==y.test])/length(y.test)
    kap<-cohen.kappa(table(pp,y.test),n.obs=length(y.test))
    blob_performance<-rbind(blob_performance,c(acc,kap$kappa))
    Anames<-c(Anames,"spLogGroupmin")
  }
  #####
  ###
  ####################
  
  #
  ##############################################
  #### wrapper functions for some classifiers
  # https://mlr-org.github.io/mlr-tutorial/devel/html/feature_selection/index.html
  
  my.task<-makeClassifTask(data = mydata[ii,], target = "Class")
  ctrl = makeFeatSelControlSequential(method = "sbs", alpha = 0.02) #backward search - change method for other searches.
  set.seed(1)
  rdesc = makeResampleDesc("CV", iters = 10) # how to evaluate
  sfeats = selectFeatures(learner = "classif.lda", task = my.task, resampling = rdesc, control = ctrl,
                          show.info = FALSE)
  sfeats
  #
  usewhich<-apply(matrix(names(x.train),length(names(x.train)),1),1,is.element,set=sfeats$x)
  #tr<-lda(x.train[,usewhich==T],y.train)
  #pp<-predict(tr,x.test[,usewhich==T])$class
  #acc<-length(y.test[pp==y.test])/length(y.test)
  #kap<-cohen.kappa(table(pp,y.test),n.obs=length(y.test))
  
  ##### Alternatively, prediction on selected features through mlr
  uu<-rep(T,dim(mydata)[2])
  uu[-classlab]<-usewhich
  my.task2<-makeClassifTask(data = mydata[,uu==T], target = "Class")
  
  lrn = makeFeatSelWrapper("classif.lda", resampling = rdesc,
                           control = makeFeatSelControlRandom(maxit = 10), show.info = FALSE)
  mod<-mlrtrain(lrn,my.task2,subset=ii)
  pp2<-predict(mod,my.task2,subset=setdiff(seq(1,dim(mydata)[1]),ii))
  #
  acc<-length(y.test[pp2$data$response==y.test])/length(y.test)
  kap<-cohen.kappa(table(pp2$data$response,y.test),n.obs=length(y.test))
  c(acc,kap$kappa)
  blob_performance<-rbind(blob_performance,c(acc,kap$kappa))
  Anames<-c(Anames,"LDA-rfe")
  FS<-rbind(FS,rep(0,length(names(x.train))))
  FS[dim(FS)[1],usewhich]<-1
  
  #### which methods can be used here?
  #lrns = listLearners()
  #head(lrns[c("class", "package")])
  #### which can be used for multiclass tasks?
  #lrns = listLearners(iris.task, properties = "prob")
  #head(lrns[c("class", "package")])
  
  ##### Results?
  rownames(blob_performance)<-Anames
  rownames(FS)<-Anames
  colnames(FS)<-names(x.train)
  return(blob_performance)
}

#### NOTE: caret and mlr have overlapping function names!
# rename the mlr train and mlrtrain
mlrtrain<-mlr::train
train<-caret::train
## https://topepo.github.io/caret/index.html
##############################################




AA.3 <- perform_analyses(3)
AA.5 <- perform_analyses(5)
AA.10 <- perform_analyses(10)
AA.15 <- perform_analyses(15)
AA.20 <- perform_analyses(20)
AA.30 <- perform_analyses(30)
