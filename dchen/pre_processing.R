#load in libraries
library(Hmisc)
library(caret)
library(glmnet)
library(randomForest)

#read in data
labels <- read.csv("~/Github/driven_pp/dchen/raw/train_labels.csv",na.string='',stringsAsFactors=F,row.names='id')
train <- read.csv("~/Github/driven_pp/dchen/raw/train_values.csv",na.string='',stringsAsFactors=F,row.names='id')
test <- read.csv("~/Github/driven_pp/dchen/raw/test_values.csv",na.string='',stringsAsFactors=F,row.names='id')
combined_x <- rbind(train,test)

#drop low-coverage columns (<80%)
combined_x<- combined_x[,apply(combined_x, 2, function(x) (length(x)-sum(is.na(x)))/length(x)) >= 0.50]
dim(combined_x)

#create model matrix 
ord_cols <- names(combined_x)[grepl('o_',names(combined_x))]
cat_cols <- names(combined_x)[grepl('c_',names(combined_x))]
num_cols <- names(combined_x)[grepl('n_',names(combined_x))]

#ordinal variables: turned ordinal variables with more than 10 different levels into numeric, otherwise categorical
num_cols <- union(ord_cols[apply(combined_x[ord_cols],2,function(x) length(unique(x))>10)],num_cols)
cat_cols <- union(ord_cols[apply(combined_x[ord_cols],2,function(x) length(unique(x))<=10)],cat_cols)
#turn all categorical variables into characters (required for model.matrix)
combined_x[cat_cols] <- lapply(combined_x[cat_cols],as.character)

# data imputation 
# mean imputation for nums
combined_x[num_cols] <- lapply(combined_x[num_cols],function(x) impute(x,mean))
# random imputation for cats 
combined_x[cat_cols] <- lapply(combined_x[cat_cols],function(x) impute(x,'random'))

#create model matrix 
fm<-as.formula(paste("~ 0+", paste(names(combined_x),collapse="+")))
combined_x <- model.matrix(fm,data=combined_x)

#drop constant columns (for PCA)
combined_x <- combined_x[,apply(combined_x, 2, var, na.rm=TRUE)!=0]
combined_x <- data.frame(combined_x)

#final combined_x: 18305 * 238

#split combined into training and test 
train_x <- merge(rownames(train),combined_x,by=0,all.x=T)
rownames(train_x) <- rownames(train)
train_x <- train_x[,dimnames(combined_x)[[2]]]

test_x <- merge(rownames(test),combined_x,by=0,all.x=T)
rownames(test_x) <- rownames(test)
test_x <- test_x[,dimnames(combined_x)[[2]]]

#preprocessing with PCA 
combinedPreProc <- preProcess(combined_x, method = 'pca')
#predict training
pca_train_x <- predict(combinedPreProc,train_x)
#predict test 
pca_test_x <- predict(combinedPreProc,test_x)

#process DVs and create separate vectors for each of them 
train_dvs <- names(labels)
dv_list <- list()
for (dv in train_dvs){
  dv_list[dv] <- labels[dv]
}


# End of pre-processing 


