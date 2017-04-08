#Link: http://blog.hackerearth.com/feature-engineering-h2o-gradient-boosting-gbm-r-scores-0-936
path <- "/home//Desktop/MLC01/"
setwd(path)

# Load data and libraries -------------------------------------------------

library(data.table)
library(h2o)
library(caret)
library(stringr)
library(e1071)

train <- fread("train_indessa.csv",na.strings = c(""," ",NA))
test <- fread("test_indessa.csv",na.strings = c(""," ",NA))

# Check Data --------------------------------------------------------------

dim(train)
dim(test)

str(train)
str(test)


# Data Preprocessing ---------------------------------------


# Check correlation and remove correlated variables -----------------------

num_col <- colnames(train)[sapply(train, is.numeric)]
num_col <- num_col[!(num_col %in% c("member_id","loan_status"))]

corrplot::corrplot(cor(train[,num_col,with=F]),method = "number")

train[,c("funded_amnt","funded_amnt_inv","collection_recovery_fee") := NULL]
test[,c("funded_amnt","funded_amnt_inv","collection_recovery_fee") := NULL]


# Extract term value and convert to integer -------------------------------

train[,term := unlist(str_extract_all(string = term,pattern = "\\d+"))]
test[,term := unlist(str_extract_all(string = term,pattern = "\\d+"))]

train[,term := as.integer(term)]
test[,term := as.integer(term)]


# Fix emp_length variable and extract values ------------------------------

train[emp_length == "n/a", emp_length := -1]
train[emp_length == "< 1 year", emp_length := 0]
train[,emp_length := unlist(str_extract_all(emp_length,pattern = "\\d+"))]
train[,emp_length := as.integer(emp_length)]

test[emp_length == "n/a", emp_length := -1]
test[emp_length == "< 1 year", emp_length := 0]
test[,emp_length := unlist(str_extract_all(emp_length,pattern = "\\d+"))]
test[,emp_length := as.integer(emp_length)]



# you can extract features out of description, I'm removing it ------------

train[,desc := NULL]
test[,desc := NULL]


# Encoding initial_list_status as 0,1 -------------------------------------

train[,initial_list_status := as.integer(as.factor(initial_list_status))-1]
test[,initial_list_status := as.integer(as.factor(initial_list_status))-1]


# fixing annual_inc variable  | skewness ----------------------------------

ggplot(train,aes(annual_inc))+geom_density(fill='lightblue',color='black')

train[is.na(annual_inc), annual_inc := 0]
train[,annual_inc := log(annual_inc + 10)]

test[is.na(annual_inc), annual_inc := 0]
test[,annual_inc := log(annual_inc + 10)]


# Check skewness of other variables ---------------------------------------

#train
se <- colnames(train)[sapply(train, is.numeric)]
se <- se[!(se %in% c("member_id","loan_status"))]

skew <- sapply(train[,se,with=F], function(x) skewness(x,na.rm = T))
skew <- skew[skew > 2] #filter variables with skewness > 2

train[,(names(skew)) := lapply(.SD, function(x) log(x + 10)), .SDcols = names(skew)]

#test
skew_t <- sapply(test[,se,with=F], function(x) skewness(x,na.rm = T))
skew_t
skew_t <- skew_t[skew_t > 2]

test[,(names(skew)) := lapply(.SD, function(x) log(x + 10)), .SDcols = names(skew)]

train[,dti := log10(dti + 10)]
test[,dti := log10(dti + 10)]


# Dropping Variables ------------------------------------------------------

train[,pymnt_plan := NULL]
test[,pymnt_plan := NULL]

train[,verification_status_joint :=  NULL]
test[,verification_status_joint :=  NULL]

train[,application_type := NULL]
test[,application_type := NULL]

train[,title := NULL]
test[,title := NULL]

train[,batch_enrolled := NULL]
test[,batch_enrolled := NULL]



# One Hot Encoding --------------------------------------------------------

train_mod <- train[,.(last_week_pay,grade,sub_grade,purpose,verification_status,home_ownership)]
test_mod <- test[,.(last_week_pay,grade,sub_grade,purpose,verification_status,home_ownership)]

# train_mod[is.na(train_mod)] <- "-1"
# test_mod[is.na(test_mod)] <- "-1"

train_ex <- model.matrix(~.+0, data = train_mod)
test_ex <- model.matrix(~.+0, data = test_mod)

train_ex <- as.data.table(train_ex)
test_ex <- as.data.table(test_ex)

dg <- setdiff(colnames(train_ex), colnames(test_ex))

train_ex <- train_ex[,-dg,with=F]

new_train <- cbind(train, train_ex)
new_test <- cbind(test, test_ex)

new_train[,c("last_week_pay","grade","sub_grade","purpose","verification_status","home_ownership") := NULL]
new_test[,c("last_week_pay","grade","sub_grade","purpose","verification_status","home_ownership") := NULL]

comb_data <- rbind(new_train,new_test,fill=TRUE)


# Encoding the addr_state,zip_code,emp_title ------------------------------

#we did not OHE these variables because of high cardinality. As these would have increased
#the data dimension manifolds and my laptop isn't that powerful

#addr_state, zip_code, emp_title
for(i in colnames(comb_data)[sapply(comb_data, is.character)])
  set(x = comb_data, j = i, value = as.integer(as.factor(comb_data[[i]])))


# Get training and test data ----------------------------------------------

F_train <- comb_data[!(is.na(loan_status))]
F_test <- comb_data[is.na(loan_status)]


# Remove files to free memory ---------------------------------------------

rm(comb_data,new_train,new_test,train_ex,test_ex)
gc()


# Feature Engineering -----------------------------------------------------

#Beyond these features there's a huge scope of new features

F_train[,new_var_2 := log(annual_inc/loan_amnt)]
F_test[,new_var_2 := log(annual_inc/loan_amnt)]

F_train[,new_var_3 := total_rec_int + total_rec_late_fee]
F_test[,new_var_3 := total_rec_int + total_rec_late_fee]

F_train[,new_var_4 := sqrt(loan_amnt * int_rate)]
F_test[,new_var_4 := sqrt(loan_amnt * int_rate)]

train[,new_var_5 := mean(loan_amnt),grade]
F_train[,new_var_5 := train$new_var_5]

xkm <- train[,mean(loan_amnt),grade]

test <- xkm[test, on="grade"]
F_test[,new_var_5 := test$V1]

train[last_week_pay == "NAth week", last_week_pay := "1000th Week"]
train[,last_week_pay_num := unlist(str_extract_all(string = last_week_pay,pattern = "\\d+"))]
train[,last_week_pay_num := as.integer(last_week_pay_num)]

test[last_week_pay == "NAth week", last_week_pay := "1000th Week"]
test[,last_week_pay_num := unlist(str_extract_all(string = last_week_pay,pattern = "\\d+"))]
test[,last_week_pay_num := as.integer(last_week_pay_num)]

F_train[,last_week_pay_num := train$last_week_pay_num]
F_test[,last_week_pay_num := test$last_week_pay_num]

F_train[,new_var_6 := ifelse(last_week_pay_num >= 155 | last_week_pay_num < 165,1,0)]
F_test[,new_var_6 := ifelse(last_week_pay_num >= 155 | last_week_pay_num < 165,1,0)]


# Machine Learning with H2o -----------------------------------------------

#Give H2o your maximum memory for computation
#if your laptop is 8GB, give atleast 6GB, close all other apps while computation happens
#may be, go out take a walk! 

h2o.init(nthreads = -1,max_mem_size = "10G") 

h2o_train <- as.h2o(F_train)
h2o_test <- as.h2o(F_test)

h2o_train$loan_status <- h2o.asfactor(h2o_train$loan_status)


# Create a validation frame -----------------------------------------------

#Here I want to avoid doing k-fold CV since data set is large, it would take longer time
#hence doing hold out validation

xd <- h2o.splitFrame(h2o_train,ratios = 0.6)

split_val <- xd[[2]]

y <- "loan_status"
x <- setdiff(colnames(F_train), c(y,"member_id"))



# Training a GBM Model ----------------------------------------------------

gbm_clf <- h2o.gbm(x = x
                     ,y = y
                     ,training_frame = h2o_train
                     ,validation_frame = split_val
                     ,ignore_const_cols = T
                     ,ntrees = 1000
                     ,max_depth = 20
                     ,stopping_rounds = 10
                     ,model_id = "gbm_model"
                     ,stopping_metric = "AUC"
                     ,learn_rate = 0.05
                     ,col_sample_rate_per_tree = 0.8
                     ,sample_rate = 0.8
                     ,learn_rate_annealing = 0.99
                     
                     
)

gbm_clf #Validation Accuracy = 0.9858

gbm_clf_pred <- as.data.table(h2o.predict(gbm_clf_3,h2o_test))
head(gbm_clf_pred,10)

sub_pred1 <- data.table(member_id = test$member_id, loan_status = gbm_clf_pred$p1)
fwrite(sub_pred1,"h2o_gbm_sub_pred1.csv") #0.936 leaderboard score

