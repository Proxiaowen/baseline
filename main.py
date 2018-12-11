# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:16:10 2018

@author: Administrator
"""
#import all the preprocession methods
from sklearn.decomposition import PCA,KernelPCA,FastICA,FactorAnalysis
from sklearn.pipeline import make_pipeline
# import all resgression methods
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest,VarianceThreshold,mutual_info_regression,f_regression
import numpy as np
import pandas as pd
from sklearn import preprocessing 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE,RandomOverSampler

def getresult(y1,scores):
    result=[0,0,0,0,0,0,0,0,0]
    for ii in range(0,len(scores)):
        if y1[ii]<=10:
            if scores[ii]<=10:
                result[0]=result[0]+1
            elif scores[ii]>=15:
                result[2]=result[2]+1
            else:
                result[1]=result[1]+1
        if y1[ii]>=15:
            if scores[ii]<=10:
                result[6]=result[6]+1
            elif scores[ii]>=15:
                result[8]=result[8]+1
            else:
                result[7]=result[7]+1
        if y1[ii]>10 and y1[ii]<15:
            if scores[ii]<=10:
                result[3]=result[3]+1
            elif scores[ii]>=15:
                result[5]=result[5]+1
            else:
                result[4]=result[4]+1
    acc=(result[0]+result[3]+result[8])/sum(result)
   
    return acc

# set all the resgression methods
RFR=RandomForestRegressor(random_state=0)
BR=BaggingRegressor(random_state=0)
GBR=GradientBoostingRegressor(random_state=0)
DTR=DecisionTreeRegressor(random_state=0)
MLPR=MLPRegressor(random_state=0)
LR=LogisticRegression(random_state=0)
SVR=SVR()
KNR=KNeighborsRegressor(weights='uniform')
ABR=AdaBoostRegressor(loss='square',random_state=0)
XGB=XGBRegressor(max_depth=5,learning_rate=0.05,n_eatimators=80,subsample=0.8,colsample_bytree=1,reg_lambda=1.5)

AllRS=[RFR,BR,GBR,DTR,MLPR,KNR,ABR,LR,SVR,XGB]
ALLPS=['MI_reg','F-regression','VAR','PCA','KernelPCA','FasterICA','FactorAnalysis']
#AllRS=[LR] 

dimenum=list(range(1,40,3))

allfeature=np.loadtxt(open(r'F:\trainfeature_cleartitle_die_name_translivetime.csv','r',encoding='utf-8-sig'),delimiter=",",skiprows=0)

resultspdf=pd.DataFrame(columns=['Regression Method','Seletion Method','Dimension Number','ACC-mean','ACC-std'])

for clf in AllRS:
    for ps in ALLPS:
        for dn in dimenum:
             
            acc=[]

            for i in range(100):
                
                if  ps =='PCA':
                    clfpine=make_pipeline(PCA(n_components=dn),clf)
                elif ps== 'KernelPCA':
                    clfpine=make_pipeline(KernelPCA(n_components=dn),clf)
                elif ps=='FasterICA':
                    clfpine=make_pipeline(FastICA(n_components=dn),clf)
                elif ps=='FactorAnalysis':
                    clfpine=make_pipeline(FactorAnalysis(n_components=dn),clf)
                elif ps=='MI_reg':
                    clfpine=make_pipeline(SelectKBest(mutual_info_regression,k=34),clf)
                elif ps=='F-regression':
                    clfpine=make_pipeline(SelectKBest(f_regression,k=34),clf)
                elif ps=='VAR':
                    clfpine=make_pipeline(VarianceThreshold(threshold=(.8 * (1 - .8))),clf)           

                # prepare the train and test data
                # prepara the train and test regression label
                train,test=train_test_split(allfeature,test_size=0.2,random_state=i)
                traindata=preprocessing.scale(train[:,:-1])
                
                trainlabel=train[:,-1] 
                testdata=preprocessing.scale(test[:,:-1])
                testlabel=test[:,-1]

                if clf==LR or clf==SVR: 
                    
                    traindata=traindata.astype(int)
                    trainlabel=trainlabel.astype(int) 
                    testdata=testdata.astype(int)
                    testlabel=testlabel.astype(int)
                
                clfpine.fit(traindata,trainlabel)
                # evaluate performance function
                testpredict=clfpine.predict(testdata)
                # compute auc acc sensitivity
                ACC=getresult(testlabel,testpredict)
                acc.append(ACC)
            resultspdf=resultspdf.append({'Regression Method':str(clf.__class__.__name__),
                                          'Seletion Method':ps,
                                          'Dimension Number':dn,
                                          'ACC-mean':np.mean(acc),
                                          'ACC-std':np.std(acc)}, ignore_index = True)
resultspdf.to_csv(r'F:\20181209parejscode\bratsbaseline.csv')
