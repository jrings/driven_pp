#Modeling: training
train.y <- dv_list[[1]]
train.x <- data.matrix(train_x)[,1:100]
fit1 <- cv.glmnet(train.x,train.y,family='binomial',alpha=1)
print(fit1)
s=fit1$lambda.min
c.fit1 <- coef(fit1,s)
names <- c.fit1@Dimnames[[1]][c.fit1@i+1]

#Prediction
test_x <- data.matrix(test_x)[,1:100]
pred = predict(fit1,newx=test_x,type='response')