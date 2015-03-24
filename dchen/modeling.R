#Model 1: PCA (top 100 PCs) + KNN (k=10) 

knn_pred_list <- list()
train.x <- data.matrix(pca_train_x)[,1:100]
test.x <- data.matrix(pca_test_x)[,1:100]

for (i in 1:length(dv_list)) {
  train.y <- as.factor(dv_list[[i]])
  knn_fit <- knn3(train.x, train.y,k=10)
  knn_pred_list[[i]] <- predict(knn_fit, test.x, k=10, type = "prob")
}

#Model 2: PCA (PCs for 95% variance explained) + NN (layers=5)
nn_pred_list <- list()
train.x <- train_x
test.x <- test_x

for (i in 1:length(dv_list)){
  train.y <- dv_list[[i]]
  nn_fit <- pcaNNet(x=train.x,y=train.y,size=5,thresh = 0.95,MaxNWts=10000)
  nn_pred_list[[i]] <- predict(nn_fit,test.x)
}


#Modeling: Random Ferns Classifier (terrible)
#library(rFerns)
# train.y is a logical matrix with each column corresponding to a class for multi-label classification
#train.y <- sapply(labels,as.logical)
#ferns_fit <- rFerns(x=train.x,y=train.y,saveForest = T)  
#pd <- predict(ferns_fit,train.x,scores=F)*1 #apply prediction directly on training set
#sum(pd[,1]==labels[,1]) #7304...

#prediction outputs written into formatted 
pred <- read.csv("~/Github/driven_pp/dchen/raw/SubmissionFormat.csv",na.string='',stringsAsFactors=F)

for (i in 1:length(dv_list)){
  dv <- names(pred)[i+1]
  pred[i+1] <- knn_pred_list[[i]][,2]
}

pred$service_d <- ifelse(pred$service_d!=0,1,0)

write.table(pred,file = 'submission_knn_10.csv',sep = ',',row.names = F)

for (i in 1:length(dv_list)){
  dv <- names(pred)[i+1]
  pred[i+1] <- nn_pred_list[[i]]
}

write.table(pred,file = 'submission_nn_5.csv',sep = ',',row.names = F)



