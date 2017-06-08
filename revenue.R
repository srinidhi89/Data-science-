

rm(list=ls(all=TRUE))

#Read data into R
setwd("E:\\INSOFE\\Projects\\Kaggle datasets")

#Train data structure and summary
train1=read.csv("train.csv",header=T,sep=",")
str(train1)
summary(train1)
sum(is.na(train1)) #check for missing values

revenue_train=train1[,43]
revenue_train1=log10(revenue_train)# log transform the target variable

#calculate age of train data records
#install.packages("eeptools")
library(eeptools)
c_date=Sys.Date()
c_date
open_date=subset(train1,select="Open.Date")
op_date=as.Date.factor(open_date$Open.Date,format = "%m/%d/%Y")
op_date
for(i in 1)
{
  age=age_calc(dob=op_date,enddate=c_date,units="years")
  print(age)
}
train1=cbind(train1,age)
train1=subset(train1,select = -c(Id,Open.Date,City,revenue))

#Test data structure and summary
test1=read.csv("test.csv",header=T,sep=",")
str(test1)
summary(test1)
sum(is.na(test1))

#calculate the age for test records
library(eeptools)
ct_date=Sys.Date()
ct_date
test_date=subset(test1,select="Open.Date")
opt_date=as.Date.factor(test_date$Open.Date,format = "%m/%d/%Y")
opt_date
for(i in 1)
{
  age=age_calc(dob=opt_date,enddate=ct_date,units="years")
  print(age)
}
test1=cbind(test1,age)
test1=subset(test1,select=-c(Id,Open.Date,City))

#combine test and train data
fin_data=rbind(train1,test1)
str(fin_data)

#adjust the levels for Type variable 
fin_data$Type[fin_data$Type == "DT"] <- "IL"
fin_data$Type[fin_data$Type == "MB"] <- "FC"

#separate numerical --> decimal values and >8 unique values and categorical data --> 4 to 8 unique values
fin_data_num=fin_data[,c(4,5,6,15,16,18,19,21,22,25,26,28,29,30,31,32,33,34,40)]
fin_data_cat=fin_data[,c(1,2,3,7,8,9,10,11,12,13,14,17,20,23,24,27,35,36,37,38,39)]
fin_data_cat=data.frame(apply(fin_data_cat,2,function(x){as.factor(x)}))
str(fin_data_cat)

#create dummies of categorical variables 
#install.packages("dummies")
library(dummies)
fin_data_cat1=dummy.data.frame(fin_data_cat, names = c(names(fin_data_cat)))

#standardize the values using range
#install.packages("vegan")
library(vegan)
fin_data_num1=decostand(fin_data_num,"range")

tot_data=cbind(fin_data_num1,fin_data_cat1)

#divide data into train and test 
set.seed(263)
train_index=tot_data[1:137,]
test_index=tot_data[138:100137,]

#build linear multivariate regression model

linregmodel=lm(revenue_train1~.,data=train_index)
summary(linregmodel)
library(MASS)
stepAIC(linregmodel)

linstep=lm(formula = revenue_train1 ~ P2 + P3 + P13 + P16 + P17 + P20 + 
     P23 + P26 + P27 + P28 + P30 + P31 + age + `City.GroupBig Cities` + 
     `P1 1` + `P1 2` + `P1 3` + `P1 4` + `P1 5` + `P1 6` + `P1 9` + 
     P51 + P52 + P53 + P54 + P55 + P56 + `P6 1` + `P6 2` + `P6 3` + 
     `P6 4` + `P6 5` + `P6 6` + `P6 8` + `P7 1` + `P7 2` + `P7 3` + 
     `P7 4` + `P8 1` + `P8 2` + `P8 3` + `P8 4` + `P8 6` + `P8 8` + 
     `P9 4` + `P9 8` + `P10 4` + `P10 8` + `P11 1` + `P11 2` + 
     `P11 3` + `P11 6` + `P11 8` + `P12 6` + `P15 0` + `P15 1` + 
     `P15 2` + `P15 3` + `P15 4` + `P15 5` + `P15 8` + `P18 1` + 
     `P18 3` + `P18 4` + `P18 5` + `P18 9` + `P21 1` + `P21 2` + 
     `P21 3` + `P21 4` + `P21 5` + P221 + P222 + P223 + P224 + 
     `P25 1` + `P25 2` + `P25 3` + `P25 4` + P332 + P334 + `P34 2` + 
     `P34 3` + `P34 4` + `P35 1` + `P35 2` + `P35 4` + `P36 3` + `P36 4` + P371 + P372 + P373 + P374, data = train_index)
summary(linstep) # R sqrd 0.78 Adj R sqrd 0.31 p-value: 0.03
lmpred=predict.lm(linstep,test_index)
lmpred=data.frame(10^lmpred)

#install.packages("DMwR")
library(DMwR)
#Error verification 
regr.eval(revenue_train, fitted(linstep))# MAPE:0.99% 

id=0:99999
id<-as.data.frame(id)
lmsubmit<-cbind(id,lmpred)
head(lmsubmit)
colnames(lmsubmit)=c("Id","Prediction")
write.csv(lmsubmit,file="LinearRegression.csv",row.names = F)

######Model 2 : PCA for dimensionality reduction & Random forest for predictions####

#PCA on train data
prin_comp <- prcomp(train_index)
names(prin_comp)
#mean of variables
prin_comp$rotation
#dimensions of matrix
dim(prin_comp$x)
#plot resulting principal components
biplot(prin_comp, scale = 0)
#standard deviation of each principal component
std_dev=prin_comp$sdev
#compute variance
pr_var=std_dev^2
#check variance of first 10 components
pr_var[1:10]
#proportion of variance explained
prop_varex <- (pr_var/sum(pr_var))*100
prop_varex[1:80]

#scree plot

plot(prop_varex, xlab = "Principal Component",ylab = "Proportion of Variance Explained",type = "b")
plot(cumsum(prop_varex), xlab = "Principal Component",ylab = "Cumulative Proportion of Variance Explained",type = "b")

#extract the 60 principal components 98.31
train_data=data.frame(prin_comp$x)
train_data=train_data[,1:60]

#transform test data into new coordinate system
test_dat=predict(prin_comp,test_index)

#build randomforest model using 60 PC
#install.packages("randomForest")
library(randomForest)
set.seed(345)
rf=randomForest(revenue_train1 ~ .,data = train_data,ntree=25,mtry=11)
summary(rf)
rev_tr_pred=rf$predicted
rev_tr_pred1=data.frame(10^rev_tr_pred)
library(DMwR)
regr.eval(revenue_train, rev_tr_pred1)
#####Output: ntree= 25 mtry=11 MAPE: 0.39

#predict using test data
rev_ts_pred=predict(rf,test_dat,type="response")
rev_ts_pred1=data.frame(10^rev_ts_pred)

############Cross Validation using training data##########
#install.packages("caret")
library(caret)
#install.packages("e1071")
library(e1071)
train_control=trainControl(method="cv", number=5)
tunegrid <- expand.grid(.mtry=c(1:10))
set.seed(789)
cv_model=train(train_data,revenue_train1,trControl=train_control,method="rf",tuneGrid=tunegrid,metric="RMSE")
summary(cv_model)
plot(cv_model)

# Predicton on cv train Data
cv_pred_model <-predict(cv_model,train_data)
cv_pred_model=data.frame(10^cv_pred_model)

# Predicton on cv Test Data
cv_pred_model_test <-predict(cv_model,test_dat)
cv_pred_model_test=data.frame(10^cv_pred_model_test)

#error metrics on cv train  data
regr.eval(revenue_train,cv_pred_model)#mape: 0.143

id=0:99999
id<-as.data.frame(id)
submit3<-cbind(id,cv_pred_model_test)
head(submit3)
colnames(submit3)=c("Id","Prediction")
write.csv(submit3,file="PCA_RF_CrossValidation_Cat.csv",row.names = F)
###Kaggle Score : 1839942.61941.

####Model 3 :Use RF for identifying important attributes and SVM for predictions####
library(randomForest)
set.seed(357)
rfmodel=randomForest(train_index,revenue_train1,ntree=25,mtry=7)
summary(rfmodel)
print(rfmodel)
rfmodel$importance
round(importance(rfmodel), 2)

# Extract and store important variables obtained from the random forest model
Imp_rf <- data.frame(rfmodel$importance)
Imp_rf <- data.frame(row.names(Imp_rf),Imp_rf[,1])
colnames(Imp_rf) = c('Attributes','Importance')
Imp_rf <- Imp_rf[order(Imp_rf$Importance , decreasing = TRUE),]
plot(Imp_rf$Attributes,Imp_rf$Importance,xlab="Attributes",ylab="Importance",main="Importance of Attributes using Random Forest")
Imp_rf <- data.frame(Imp_rf[1:65,]) ## >=0.01
name1=(Imp_rf$Attributes[1:65])

train_index_svm=subset(tot_data,select=name1)
tr_svm=train_index_svm[1:137,]
ts_svm=train_index_svm[138:100137,]

library(e1071)
mod_reg <- svm(x = tr_svm, y = revenue_train1,type = 'nu-regression', kernel = 'linear')
pred_svm=predict(mod_reg,ts_svm)
pred_svm=data.frame(10^pred_svm)
pred_svm

#error metrics on train data
regr.eval(revenue_train,pred_svm)#mape : 4.13e+02 


#######Cross Validation using Train Data#########
#install.packages("caret")
library(caret)
#install.packages("e1071")
library(e1071)
#install.packages("kernlab")
library(kernlab)
set.seed(782)
tr_control=trainControl(method="cv", number=5)
gridsvm <- expand.grid(C = c(10,2.5,0.1,0.01,0.2,1.5))
cv_svm=train(tr_svm,revenue_train1,trControl=tr_control,method="svmLinear",tuneGrid=gridsvm,metric="RMSE",tuneLength = 6)
summary(cv_svm)
plot(cv_svm)

# Predicton on cv train Data
cv_svm_model <-predict(cv_svm,tr_svm)
cv_svm_model=data.frame(10^cv_svm_model)

# Predicton on cv Test Data
cv_svm_test <-predict(cv_svm,ts_svm)
cv_svm_test=data.frame(10^cv_svm_test)

#error metrics on cv train  data
regr.eval(revenue_train,cv_svm_test)#mape: 2.92*10^2


id=0:99999
id<-as.data.frame(id)
submit4<-cbind(id,cv_svm_test)
head(submit4)
colnames(submit4)=c("Id","Prediction")
write.csv(submit4,file = "RF_SVM_Predictions_Cat.csv",row.names = F)
#####Kaggle Score: 1891545.57105.

####Model 4:Convert integers into categorical data and combine with numeric data build a random forest####

str(tot_data)
train_rows=tot_data[1:137,]
test_rows=tot_data[138:100137,]

library(randomForest)
set.seed(215)
rfmodel_reg=randomForest(train_rows,revenue_train1,ntree=25,mtry=14)
summary(rfmodel_reg)
print(rfmodel_reg)
#Tuning ntree= 25,mtry=5 MAPE : 0.24,mtry=6 0.22,mtry= 7 0.21,mtry= 8 0.20,mtry= 9 0.20,mtry= 11 0.19,mtry= 13 0.18,mtry= 14 0.17

# Predicton train Data
rf_pred_model <-predict(rfmodel_reg,train_rows)
rf_pred_model=data.frame(10^rf_pred_model)

# Predicton Test Data
rf_pred_model_test <-predict(rfmodel_reg,test_rows)
rf_pred_model_test=data.frame(10^rf_pred_model_test)

#error metrics on train data
regr.eval(revenue_train, rf_pred_model)# MAPE 0.178

#############cross validation using RF###############
#install.packages("caret")
library(caret)
#install.packages("e1071")
library(e1071)
set.seed(817)
tr_ctrl=trainControl(method="cv", number=5)
gridrf <- expand.grid(.mtry=c(1:18))
cv_rf=train(train_rows,revenue_train1,trControl=tr_ctrl,method="rf",tuneGrid=gridrf,metric="RMSE")
summary(cv_rf)
plot(cv_rf)
#mtry= 10 MAPE : 0.193 ,mtry= 12 0.189 ,mtry=18 0.172 mtry= 20 0.175

#Predicton on cv train Data
cv_rf_md <-predict(cv_rf,train_rows)
cv_rf_md=data.frame(10^cv_rf_md)

#Predicton on cv Test Data
cv_rf_ts <-predict(cv_rf,test_rows)
cv_rf_ts=data.frame(10^cv_rf_ts)

#error metrics on cv train  data
regr.eval(revenue_train,cv_rf_md)#mape: 0.172

id=0:99999
id<-as.data.frame(id)
submit5<-cbind(id,cv_rf_ts)
head(submit5)
colnames(submit5)=c("Id","Prediction")
write.csv(submit5,file="RF_Test_Cat.csv",row.names = F)
##Kaggle Score : 1697423.48773

############Neural Net for Regression############

library(nnet)
set.seed(218)
nnmodel_reg=nnet(train_data,revenue_train1,size = 3, decay = 5e-2, maxit = 70,linout=T)
summary(nnmodel_reg)
#Tuning size=4 decay= 0.05,maxit =50 MAPE 0.132, [3,0.05,70] --0.12

# Predicton train Data
nn_pred_model <-predict(nnmodel_reg,train_data)
nn_pred_model=data.frame(10^nn_pred_model)

# Predicton Test Data
nn_pred_model_test <-predict(nnmodel_reg,test_dat)
nn_pred_model_test=data.frame(10^nn_pred_model_test)

#error metrics on train data
regr.eval(revenue_train, nn_pred_model)# MAPE 0.120

#############cross validation using NNET###############
#install.packages("caret")
library(caret)
#install.packages("nnet")
library(nnet)
set.seed(819)
tr_ctr=trainControl(method="cv", number=5)
gridnn <- expand.grid(.decay=c(0.02,0.1,0.01,0.001,0.0004,0), .size=c(3,4))
cv_nn=train(train_data,revenue_train1,trControl=tr_ctr,method="nnet",tuneGrid=gridnn,metric="RMSE",linout=T,maxit=80)
summary(cv_nn)
plot(cv_nn)

#Predicton on cv train Data
cv_nn_md <-predict(cv_nn,train_data)
cv_nn_md=data.frame(10^cv_nn_md)

#Predicton on cv Test Data
cv_nn_ts <-predict(cv_nn,test_dat)
cv_nn_ts=data.frame(10^cv_nn_ts)

#error metrics on cv train  data
regr.eval(revenue_train,cv_nn_md)#MAPE: 0.117

id=0:99999
id<-as.data.frame(id)
submit6<-cbind(id,cv_nn_ts)
head(submit6)
colnames(submit6)=c("Id","Prediction")
write.csv(submit6,file="NeuralNw_Test_Cat.csv",row.names = F)
##Kaggle Score : 2515645

##############Ensemble Model################


finalpred=(0.1*cv_nn_ts)+(0.4*cv_rf_ts)+(0.2*cv_svm_test)+(0.3*cv_pred_model_test)

id=0:99999
id<-as.data.frame(id)
submit7<-cbind(id,finalpred)
head(submit7)
colnames(submit7)=c("Id","Prediction")
write.csv(submit7,file="Ensemble_Test_Cat.csv",row.names = F)
###kaggle score: 1767923 


####Visualizations######

install.packages("ggplot2")
library(ggplot2) 

qplot(revenue_train1, geom="histogram",binwidth=0.1,xlab="revenue",fill=I("orange"),col=I("black"))
qplot(train1$Type)
qplot(test1$Type)

