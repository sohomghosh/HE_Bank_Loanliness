import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, metrics, ensemble
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
train=pd.read_csv("train_indessa.csv")
test=pd.read_csv("test_indessa.csv")

train.head()
list(train.columns)

train.loan_status.value_counts() #0    406601; 1    125827
###MAY NEED TO BALANCE 0's & 1'S


test['loan_status']='NA'
train_test = train.append(test)
train_test.describe()

'''
train_test['Gender'] = train_test['Gender'].replace(to_replace = {'Male': 0, 'Female': 1})
type_of_cab = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
train_test['Type_of_Cab'] = train_test['Type_of_Cab'].replace(to_replace = type_of_cab)
'''

train_test = train_test.drop('member_id', 1) #Removing member_id column

#pd.isnull(train_test['loan_amnt']).value_counts()
#train_test.iloc[0] #Slicing by row

for i in range(0,train_test.shape[1]):
	#print(str(list(train_test.columns)[i])+": " +str(pd.isnull(train_test.ix[:,i]).value_counts()))
	if len(pd.isnull(train_test.ix[:,i]).value_counts()) ==2:
		print (str(list(train_test.columns)[i]))



###Having NA values
'''
batch_enrolled
emp_title
annual_inc
desc
title
delinq_2yrs
inq_last_6mths
mths_since_last_delinq
mths_since_last_record
open_acc
pub_rec
revol_util
total_acc
collections_12_mths_ex_med
mths_since_last_major_derog
verification_status_joint
acc_now_delinq
tot_coll_amt
tot_cur_bal
total_rev_hi_lim
'''

train_test['term']=[int(str(i)[:-7]) for i in train_test['term']]
train_test.dtypes

'''
###train_test['batch_enrolled'].value_counts()
BAT2252229     18791
BAT3873588     17839
BAT2803411     17111
BAT2078974     14859
list(train_test['batch_enrolled'].unique())

###train_test['grade']
354949    A
354950    A

###sub_grade
354947    G3
354948    E4
354949    A4
354950    A4

###emp_title
354947                  Credit Risk Analyst
354948                     USPS/Nashua L&DC
354949                    Computer Engineer

###emp_length
354947       1 year
354948    10+ years
354949     < 1 year
354950      2 years

354946    MORTGAGE
354947         OWN
354948         OWN
354949    MORTGAGE
354950         OWN
Name: home_ownership, dtype: object


354947           Verified
354948           Verified
354949    Source Verified
354950       Not Verified
Name: verification_status, dtype: object


354946    n
354947    n
354948    n
354949    n
354950    n
Name: pymnt_plan, dtype: object


354946                                                  NaN
354947                                                  NaN
354948    I will be using the loan money to pay off the ...
354949                                                  NaN
354950       > This loan will be used to pay for another...
Name: desc, dtype: object


354947    debt_consolidation
354948      home_improvement
354949           credit_card
354950    debt_consolidation
Name: purpose, dtype: object


354946            Credit card refinancing
354947                 Debt consolidation
354948                 Debt Consolidation
354949            Credit card refinancing
354950            Debt Consolidation Loan
Name: title, dtype: object


354946    761xx
354947    115xx
354948    038xx
354949    206xx
354950    323xx
Name: zip_code, dtype: object


354946    TX
354947    NY
354948    NH
354949    MD
354950    FL
Name: addr_state, dtype: object


354947    f
354948    f
354949    w
354950    f
Name: initial_list_status, dtype: object


354945    INDIVIDUAL
354946    INDIVIDUAL
354947    INDIVIDUAL
354948    INDIVIDUAL
354949    INDIVIDUAL
354950    INDIVIDUAL
Name: application_type, dtype: object


354946    NaN
354947    NaN
354948    NaN
354949    NaN
354950    NaN
Name: verification_status_joint, dtype: object


354946     83th week
354947     39th week
354948     87th week
354949     35th week
354950    104th week
Name: last_week_pay, dtype: object

'''


for f in ['batch_enrolled']:#Add all categorical features in the list
    lbl = LabelEncoder()
    lbl.fit(list(train_test[f].values))
    train_test[f] = lbl.transform(list(train_test[f].values))

for f in ['grade','sub_grade','emp_title','emp_length','home_ownership','verification_status','pymnt_plan','desc','purpose','title','zip_code','addr_state','initial_list_status','application_type','verification_status_joint','last_week_pay']:
    lbl = LabelEncoder()
    lbl.fit(list(train_test[f].values))
    train_test[f] = lbl.transform(list(train_test[f].values))

for f in ['initial_list_status']:
	lbl = LabelEncoder()
	lbl.fit(list(train_test[f].values))
	train_test[f] = lbl.transform(list(train_test[f].values))

for f in ['application_type']:
	lbl = LabelEncoder()
	lbl.fit(list(train_test[f].values))
	train_test[f] = lbl.transform(list(train_test[f].values))

for f in ['verification_status_joint']:	
	lbl = LabelEncoder()
	lbl.fit(list(train_test[f].values))
	train_test[f] = lbl.transform(list(train_test[f].values))

for f in ['last_week_pay']:
	lbl = LabelEncoder()
	lbl.fit(list(train_test[f].values))
	train_test[f] = lbl.transform(list(train_test[f].values))

features = np.setdiff1d(train_test.columns, ['emp_title', 'desc','purpose','title','zip_code','addr_state','loan_status'])

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 3,
                "seed": 2016, "tree_method": "exact"}
X_train=train_test[0:len(train.index)]
X_test=train_test[len(train.index):len(train_test.index)]

dtrain = xgb.DMatrix(X_train[features], X_train['loan_status'], missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds = bst.predict(dtest)

submit = pd.DataFrame({'member_id': test['member_id'], 'loan_status': test_preds})
submit.to_csv("XGB.csv", index=False)


import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
# initialize an h2o cluster
h2o.init()
h2o.connect()
train = h2o.import_file("train_indessa.csv")
test = h2o.import_file("test_indessa.csv")

# all values of r under .8 are assigned to 'train_split' (80 percent)
train['batch_enrolled']=train['batch_enrolled'].asfactor()

for i in ['grade','sub_grade','emp_title','emp_length','home_ownership','verification_status','pymnt_plan','desc','purpose','title','zip_code','addr_state','initial_list_status','application_type','verification_status_joint','last_week_pay']:
	train[i]=train[i].asfactor()
train["loan_status"]=train["loan_status"].asfactor()

r = train.runif()   
train_split = train[r  < 0.8]
# all values of r equal or above .8 are assigned 'valid_split' (20 percent)
valid_split = train[r >= 0.8]
#train_split["loan_status"] = train_split["loan_status"].asfactor()
#valid_split["loan_status"] = valid_split["label"].asfactor()
model = H2ODeepLearningEstimator(
        distribution="multinomial",
        activation="RectifierWithDropout",
        hidden=[100,200,100],
        input_dropout_ratio=0.2, 
        sparse=True, 
        l1=1e-5, 
        epochs=100)
features = list(np.setdiff1d(train.names, ['emp_title', 'desc','purpose','title','zip_code','addr_state','loan_status']))
model.train(
        x= features, 
        y="loan_status", 
        training_frame=train_split, 
        validation_frame=valid_split)
model.params
model
pred = model.predict(test)
pred.head()
#submit_pred = pred[:,0]
submit_pred= pred[:,1]
submit_pred.head()
#submission_dataframe = h2o.H2OFrame({'member_id': list(test['member_id'])}).cbind(submit_pred)
submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submission_dataframe.set_name(1,"loan_status")
h2o.h2o.export_file(submission_dataframe, path ="submission_h20_1.csv")

'''
# set the parameters of your Deep Learning model

# train your model on your train_split data set, and then validate it on the valid_split set
# x is the list of column names excluding the response column (all of your features)
# y is the name of the response column ('label')



# Take a look at the predictions


# Rename the second column (formerely 'Predict') to 'Label'
submission_dataframe.set_name(1,"Label")






from h2o.estimators.gbm import H2OGradientBoostingEstimator
model = H2OGradientBoostingEstimator(distribution='bernoulli',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model.train(x=x, y=y, training_frame=train, validation_frame=test)                                    
print(model)
perf = model.model_performance(test)
perf.auc()
cvmodel = H2OGradientBoostingEstimator(distribution='bernoulli',
                                       ntrees=100,
                                       max_depth=4,
                                       learn_rate=0.1,
                                       nfolds=5)

cvmodel.train(x=x, y=y, training_frame=data)
ntrees_opt = [5,50,100]
max_depth_opt = [2,3,5]
learn_rate_opt = [0.1,0.2]

hyper_params = {'ntrees': ntrees_opt, 
                'max_depth': max_depth_opt,
                'learn_rate': learn_rate_opt}
from h2o.grid.grid_search import H2OGridSearch

gs = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params = hyper_params)

gs.train(x=x, y=y, training_frame=train, validation_frame=test)
print(gs)

for g in gs:
    print(g.model_id + " auc: " + str(g.auc()))

                    







r = crimes["Arrest"].runif(1234)
train = crimes[r < 0.8]
test = crimes[r >= 0.8]

# Simple GBM - Predict Arrest
crimes_names_x = crimes.names[:]
crimes_names_x.remove("Arrest")
data_gbm = H2OGradientBoostingEstimator(ntrees         =10,
                                        max_depth      =6,
                                        distribution   ="bernoulli")

data_gbm.train(x               =crimes_names_x,
               y               ="Arrest",
               training_frame  =train,
               validation_frame=test)


# GBM performance on train/test data
train_auc_gbm = data_gbm.model_performance(train).auc()
test_auc_gbm  = data_gbm.model_performance(test) .auc()

# Deep Learning performance on train/test data
# train_auc_dl = data_dl.model_performance(train).auc()
# test_auc_dl  = data_dl.model_performance(test) .auc()

# Make a pretty HTML table printout of the results
header = ["Model", "AUC Train", "AUC Test"]
table  = [
           ["GBM", train_auc_gbm, test_auc_gbm],
#            ["DL ", train_auc_dl, test_auc_dl]
         ]
h2o.display.H2ODisplay(table, header)

gbm_pred = data_gbm.predict(crime_examples)







import h2o
import time
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
h2o.init()

grouped = data.group_by(["Days","start station name"])
bpd = grouped.count().get_frame() # Compute bikes-per-day
bpd.set_name(2,"bikes")
bpd.show()
bpd.describe()
bpd.dim

print("Quantiles of bikes-per-day")
bpd["bikes"].quantile().show()


secs = bpd["Days"]*secsPerDay
bpd["Month"]     = secs.month().asfactor()
# Add in day-of-week (work-week; more bike rides on Sunday than Monday)
bpd["DayOfWeek"] = secs.dayOfWeek()
print("Bikes-Per-Day")
bpd.describe()


def split_fit_predict(data):
  global gbm0,drf0,glm0,dl0
  # Classic Test/Train split
  r = data['Days'].runif()   # Random UNIForm numbers, one per row
  train = data[  r  < 0.6]
  test  = data[(0.6 <= r) & (r < 0.9)]
  hold  = data[ 0.9 <= r ]
  print("Training data has",train.ncol,"columns and",train.nrow,"rows, test has",test.nrow,"rows, holdout has",hold.nrow)
  bike_names_x = data.names
  bike_names_x.remove("bikes")
  
  # Run GBM
  s = time.time()
  
  gbm0 = H2OGradientBoostingEstimator(ntrees=500, # 500 works well
                                      max_depth=6,
                                      learn_rate=0.1)
    

  gbm0.train(x               =bike_names_x,
             y               ="bikes",
             training_frame  =train,
             validation_frame=test)

  gbm_elapsed = time.time() - s

  # Run DRF
  s = time.time()
    
  drf0 = H2ORandomForestEstimator(ntrees=250, max_depth=30)

  drf0.train(x               =bike_names_x,
             y               ="bikes",
             training_frame  =train,
             validation_frame=test)
    
  drf_elapsed = time.time() - s 
    
    
  # Run GLM
  if "WC1" in bike_names_x: bike_names_x.remove("WC1")
  s = time.time()

  glm0 = H2OGeneralizedLinearEstimator(Lambda=[1e-5], family="poisson")
    
  glm0.train(x               =bike_names_x,
             y               ="bikes",
             training_frame  =train,
             validation_frame=test)

  glm_elapsed = time.time() - s
  
  # Run DL
  s = time.time()

  dl0 = H2ODeepLearningEstimator(hidden=[50,50,50,50], epochs=50)
    
  dl0.train(x               =bike_names_x,
            y               ="bikes",
            training_frame  =train,
            validation_frame=test)
    
  dl_elapsed = time.time() - s
  
  # ----------
  # 4- Score on holdout set & report
  train_mse_gbm = gbm0.model_performance(train).mse()
  test_mse_gbm  = gbm0.model_performance(test ).mse()
  hold_mse_gbm  = gbm0.model_performance(hold ).mse()
#   print "GBM mse TRAIN=",train_mse_gbm,", mse TEST=",test_mse_gbm,", mse HOLDOUT=",hold_mse_gbm
  
  train_mse_drf = drf0.model_performance(train).mse()
  test_mse_drf  = drf0.model_performance(test ).mse()
  hold_mse_drf  = drf0.model_performance(hold ).mse()
#   print "DRF mse TRAIN=",train_mse_drf,", mse TEST=",test_mse_drf,", mse HOLDOUT=",hold_mse_drf
  
  train_mse_glm = glm0.model_performance(train).mse()
  test_mse_glm  = glm0.model_performance(test ).mse()
  hold_mse_glm  = glm0.model_performance(hold ).mse()
#   print "GLM mse TRAIN=",train_mse_glm,", mse TEST=",test_mse_glm,", mse HOLDOUT=",hold_mse_glm
    
  train_mse_dl = dl0.model_performance(train).mse()
  test_mse_dl  = dl0.model_performance(test ).mse()
  hold_mse_dl  = dl0.model_performance(hold ).mse()
#   print " DL mse TRAIN=",train_mse_dl,", mse TEST=",test_mse_dl,", mse HOLDOUT=",hold_mse_dl
    
  # make a pretty HTML table printout of the results

  header = ["Model", "mse TRAIN", "mse TEST", "mse HOLDOUT", "Model Training Time (s)"]
  table  = [
            ["GBM", train_mse_gbm, test_mse_gbm, hold_mse_gbm, round(gbm_elapsed,3)],
            ["DRF", train_mse_drf, test_mse_drf, hold_mse_drf, round(drf_elapsed,3)],
            ["GLM", train_mse_glm, test_mse_glm, hold_mse_glm, round(glm_elapsed,3)],
            ["DL ", train_mse_dl,  test_mse_dl,  hold_mse_dl , round(dl_elapsed,3) ],
           ]
  h2o.display.H2ODisplay(table,header)


 split_fit_predict(bpd)
 
 wthr1 = h2o.import_file(path=[mylocate("bigdata/laptop/citibike-nyc/31081_New_York_City__Hourly_2013.csv"),
                               mylocate("bigdata/laptop/citibike-nyc/31081_New_York_City__Hourly_2014.csv")])
# Peek at the data
wthr1.describe()

wthr2 = wthr1[["Year Local","Month Local","Day Local","Hour Local","Dew Point (C)","Humidity Fraction","Precipitation One Hour (mm)","Temperature (C)","Weather Code 1/ Description"]]

wthr2.set_name(wthr2.names.index("Precipitation One Hour (mm)"), "Rain (mm)")
wthr2.set_name(wthr2.names.index("Weather Code 1/ Description"), "WC1")
wthr2.describe()

split_fit_predict(bpd_with_weather)

'''