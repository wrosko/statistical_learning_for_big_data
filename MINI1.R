library(readr)
library(factoextra)
library(NbClust)
library(mclust)
library(xtable)
library(ggplot2)

# setwd('d:/chalmers/statistical_learning_for_big_data')
studData <- read_csv("data/turkiye-student-evaluation_generic.csv")
mydata = studData

# Clean data, make sure it's numeric
mydata = as.data.frame(unclass(mydata))
summary(mydata)
myCleanData = na.omit(mydata)
summary(myCleanData)

myCleanData$instr <- NULL
myCleanData$class <- NULL

myCleanData2 <- myCleanData
myCleanData2$nb.repeat <- NULL
myCleanData2$attendance <- NULL
myCleanData2$difficulty <- NULL

# Scale/normalize data
scaled_data = as.matrix(scale(myCleanData2))

plt1 <- fviz_nbclust(scaled_data,kmeans,k.max = 14, method = "wss")+
  geom_vline(xintercept = 3, linetype = 2)+
  labs(subtitle = "Elbow method")
plt2 <- fviz_nbclust(scaled_data, kmeans, method = "silhouette", print.summary = TRUE)+
  labs(subtitle = "Silhouette method")
plt3 <- fviz_nbclust(scaled_data, kmeans, nstart = 25,  method = "gap_stat",k.max = 14, nboot = 5)+
  labs(subtitle = "Gap statistic method")

multiplot(plt1, plt2, plt3, cols=2)

nb <- NbClust(scaled_data, distance = "euclidean",min.nc =2,
              max.nc = 14, method = "kmeans")

nb2 <- NbClust(scaled_data, distance = "manhattan",min.nc =2,
               max.nc = 14, method = "kmeans")

hist(nb2$Best.nc[1,], breaks = max(na.omit(nb2$Best.nc[1,])))

d_clust <- Mclust(as.matrix(scaled_data), G=1:15, 
                  modelNames = mclust.options("emModelNames"))
d_clust$BIC
plot(d_clust)
