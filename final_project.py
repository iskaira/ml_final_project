#Tilegen Kairat 
#3EN03B

#Kid Creative Dataset, making prediction about people who will purchase items from store or not.
#1 is True, 0 is False in Dataset
#Dataset Had 18 Columns,the first column contained only the current number of row. Then we Got 17
#Features 17: 'Buy','Income','Is Female','Is Married','Has College','Is Professional','Is Retired','Unemployed','Residence Length','Dual Income','Minors','Own','House','White','English','Prev Child Mag','Prev Parent Mag'
#Number of Samples 673

#1. Importing all libraries
#from matplotlib.ticker import MultipleLocator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from show_confusion_matrix import show_confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from pandas_confusion import ConfusionMatrix
import pandas
from pandas.tools.plotting import scatter_matrix
import random
import statsmodels.api as sm
import time
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pylab as pl
from sklearn.neural_network import MLPClassifier
from threading import Thread
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import itertools

#Reading CSV file using pandas read_csv method
filename = '/users/kairat/Desktop/Final_Project/KidCreative.csv'
features=['Buy','Income','Is_Female','Is_Married','Has_College','Is_Professional','Is_Retired','Unemployed','Residence_Length','Dual_Income','Minors','Own','House','White','English','Prev_Child_Mag','Prev_Parent_Mag']

csv=pd.read_csv(filename,sep=',')
datasets=csv.as_matrix()
#print datasets 
dataset=[]
target=[]
data=[]
for i in range(0,len(datasets)):
    data.append([])
    dataset.append([])
    for j in range (len(datasets[i])):
        if j==0:
            continue
        else:
            dataset[i].append(datasets[i][j])
            if j==1:
                target.append(datasets[i][j])
            else:
                data[i].append(datasets[i][j])
#Dividing Dataset into Features, X(Data), Y(Target) and Applying Normalization for Dataset
#Normalization is important for making strong Variance, and to fasten analysis. Maximize the variance of each component.
dataset=np.asarray(dataset)
#dataset = dataset / np.linalg.norm(dataset)

X=np.asarray(data)
Y=np.asarray(target)

#Dividing dataset into 3 pieces: 1)Test 20% 2)Validation 20% 3)60% Training
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)#20% Test, 80%Train
X_train,X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.25,random_state=3)#20% Validation 60%Train

#Using pandas, applying dataset into pandas' dataset
frame =pd.DataFrame(dataset)
frame.columns=features

#NORMALIZATION OF PD DATASET
cols_to_norm = ['Income','Residence_Length']
frame[cols_to_norm] = frame[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

columns=[(frame.Buy),(frame.Income),(frame.Is_Female),(frame.Is_Married),(frame.Has_College),(frame.Is_Professional),(frame.Is_Retired),(frame.Unemployed),(frame.Residence_Length),(frame.Dual_Income),(frame.Minors),(frame.Own),(frame.House),(frame.White),(frame.English),(frame.Prev_Child_Mag),(frame.Prev_Parent_Mag)]
forplot=[]
column=[]
row=[]

k=0
for i in range(0,len(columns)):
    for j in range(i+1,len(columns)-1):
        if columns[i].corr(columns[j])>0.6 or columns[i].corr(columns[j])<-0.6:
            forplot.append(columns[i].corr(columns[j]))
            column.append(features[i])
            row.append(features[j])
            k+=1
def get_correlations():
    for i in range(0,len(columns)):
        for j in range(i+1,len(columns)-1):
            print 'corr btw',features[i],'and',features[j],columns[i].corr(columns[j])

'''
Correlation gives an indication of how related the changes are between two variables.
If two variables change in the same direction they are positively correlated. 
If the change in opposite directions together (one goes up, one goes down), then they are negatively correlated.
'''
def draw_high_cor():
    fig = plt.figure(figsize=(45, 15))
    plots = len(forplot)
    ax=[]
    s=0
    f=0
    for i in range(0,plots):
            ax.append(plt.subplot2grid((5,4), (s,f)))
            f+=1
            ax[i].scatter(frame[row[i]],frame[column[i]],  s=10, c=[random.random(),random.random(),random.random()], marker="o")
            ax[i].set_ylabel(column[i])
            ax[i].set_xlabel(row[i])
            if (i+1)%4==0:
                s+=1
                f=0
    plt.show()
    plt.close(fig)
'''
This is useful to know, because some machine learning algorithms 
like linear and logistic regression can have poor performance 
if there are highly correlated input variables in your data.
'''
#Draw corr figure
def correlation_fig():
    correlations = frame.corr()
    sm.graphics.plot_corr(correlations, xnames=features,ynames=features)
    plt.show()

#Show The Scatter_matrix Figure
def scatter_matrix_fig():
    scatter_matrix(frame,alpha=0.5, figsize=(10, 10), diagonal='kde')
    plt.show()

#2nd Version to show scatter_matrix
def scatter_matrix_fig2():
    sns.set()
    sns.pairplot(frame,diag_kind="kde",palette="husl")
    plt.show()

'''
Scatter plots are useful for spotting structured relationships between variables, 
like whether you could summarize the relationship between two variables with a line. 
Attributes with structured relationships may also be correlated and good candidates for removal from your dataset.
'''

#Get Histogram of Dataset

'''
Histograms group data into bins and provide you a count of the number of observations in each bin.
From the shape of the bins you can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution.
It can also help you see possible outliers.
'''
def hist_fig():
    frame.hist()
    plt.show()


#Bayes
nb=GaussianNB()
nb.fit(X_train,y_train)
nbpred=[]
#KNN
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knnpred=[]
#DT
model = DecisionTreeClassifier(min_samples_split=5)
model.fit(X_train, y_train)
dtpred=[]
#LR
logit = LogisticRegression()
logit.fit(X_train,y_train)
logitpred=[]

#SVM
svc = SVC(kernel='rbf')#for exp linear 
svc.fit(X_train,y_train)
svcpred=[]

#ANN
ann = MLPClassifier()#solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=9
ann.fit(X_train,y_train)
annpred=[]
data_arr=list(X_val)

for i in range(0,len(data_arr)):
    knnpred.append(knn.predict([data_arr[i]]))
    dtpred.append(model.predict([data_arr[i]]))
    nbpred.append(nb.predict([data_arr[i]]))
    logitpred.append(logit.predict([data_arr[i]]))
    svcpred.append(svc.predict([data_arr[i]]))
    annpred.append(ann.predict([data_arr[i]]))

# Get the general accuracy for the Default Model, this we will need in order to see the changes in new model accuracy
def general_accuracy():
    print "accuracy KNN Algorithm:",accuracy_score(y_val, knnpred)
    print "accuracy Data Tree:",accuracy_score(y_val, dtpred)
    print "accuracy Gaussian Normal:",accuracy_score(y_val,nbpred)
    print "accuracy Logistic Regression:",accuracy_score(y_val, logitpred)
    print "accuracy SVM :",accuracy_score(y_val, svcpred)
    print "accuracy ANN :",accuracy_score(y_val, annpred)

#Calculate TP,TN,FP,FN
def get_conf(predicted):
    tn, fp, fn, tp = confusion_matrix(y_val, predicted).ravel()
    print 'True positives:',tp,'\nTrue negatives:',tn,'\nFalse negatives:',fn,'\nFalse positives',fp
    print(classification_report(np.asarray(y_val), np.asarray(predicted)))
    print '********************'

#DRAW CONFUSION MATRIX FOR KNN and Statistics
def confusion_matrix_class_k():
    predicted=knn.predict(X_val)
    cm = ConfusionMatrix(y_val, predicted)
    C = confusion_matrix(y_val,predicted)
    show_confusion_matrix(C, ['Class 0', 'Class 1'],'KNN')
    cm.print_stats()
    get_conf(predicted)
#DRAW CONFUSION MATRIX FOR Logit and Statistics
def confusion_matrix_class_l():
    predicted=logit.predict(X_val)
    cm = ConfusionMatrix(y_val, predicted)  
    C = confusion_matrix(y_val,predicted)
    show_confusion_matrix(C, ['Class 0', 'Class 1'],'Logit')
    cm.print_stats()
    get_conf(predicted)
    
#as we see from predicted accuracies,
#DT was the best one, but in real time, 
#logit should be the best one, because it's yes/no problem. 
#And NB is not suitible for this


#First we Create Model According to the trained intances, and Check accuracy of data from Validation

    
def model_implementation():
    k_range=range(1,41) #number of neighbors are going to be 1 to 40
    k_scores=[] #store each mean accuracy for each neigbors number value
    p_name=['Value of K for KNN','Value of C in Logit','Value of Max iterations for Logit','Value of Max_depth for Decition Tree','Value of alpha for ANN','Value of C for SVM']
    Max_range=pl.frange(0,200,5)#testing Max_iter for Logistic
    C_range=pl.frange(0.1,1,0.1)
    n_folds=10
    C_scores=[]#store each C's accuracy for each C number value
    Max_scores=[]#store iteration
    scores_stds=[]
    scores_std=[]

    p_i=[]#parameter i (k_scores,c_scores...)
    p_j=[]# parameter j (k_range,c_range..)
    for k in k_range:
        knn2 = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn2, X_train, y_train, cv=10)
        k_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    k_scores, scores_std = np.array(k_scores), np.array(scores_std)
    p_i.append(k_scores)
    p_j.append(k_range)
    scores_std=[]
    for c in C_range:
        log = LogisticRegression(C=c)
        scores = cross_val_score(log, X_train, y_train, cv=10)
        #print scores
        C_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    C_scores, scores_std = np.array(C_scores), np.array(scores_std)    
    p_i.append(C_scores)
    p_j.append(C_range)
    
    scores_std=[]
    for M in Max_range:
        log = LogisticRegression(max_iter=M)
        scores = cross_val_score(log, X_train, y_train, cv=10)
        Max_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    Max_scores, scores_std = np.array(Max_scores), np.array(scores_std)
    p_i.append(Max_scores)
    p_j.append(Max_range)
    
    #Tree
    tree_scores=[]
    tree_range=range(3,10)
    scores_std=[]
    for M in tree_range:
        dt = DecisionTreeClassifier(max_depth=M)
        scores = cross_val_score(dt, X_train, y_train, cv=10)
        tree_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    tree_scores, scores_std = np.array(tree_scores), np.array(scores_std)
    p_i.append(tree_scores)
    p_j.append(tree_range)
    

    #ANN
    ann_scores=[]
    ann_range=pl.frange(0.0001,1,0.01)
    scores_std=[]
    for M in ann_range:
        Ann = MLPClassifier(alpha=M)
        scores = cross_val_score(Ann, X_train, y_train, cv=10)
        ann_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    ann_scores, scores_std = np.array(ann_scores), np.array(scores_std)
    p_i.append(ann_scores)
    p_j.append(ann_range)
    

    #CVM
    cvm_scores=[]
    cvm_range=pl.frange(0.1,10,0.1)
    scores_std=[]
    for M in cvm_range:
        Cvm = SVC(C=M)
        scores = cross_val_score(Cvm, X_train, y_train, cv=10)
        cvm_scores.append(scores.mean())
        scores_std.append(scores.std()*2)
    scores_stds.append(scores_std)
    cvm_scores, scores_std = np.array(cvm_scores), np.array(scores_std)
    p_i.append(cvm_scores)
    p_j.append(cvm_range)



    plt.figure(figsize=(45, 20))
    ax=[]
    s=0
    f=0
    for i in range(0,len(p_i)):
        ax.append(plt.subplot2grid((5,4), (s,f)))
        f+=1
        ax[i].semilogx(p_j[i], p_i[i],color='red')
        std_error = scores_stds[i] / np.sqrt(n_folds)
        ax[i].semilogx(p_j[i], p_i[i] + std_error, 'b--')
        ax[i].semilogx(p_j[i], p_i[i] - std_error, 'b--')
        #ax[0].scatter(p_j[i],p_i[i],  s=10, c=[random.random(),random.random(),random.random()], marker="o")
        #ax[0].plot(p_j[i],p_i[i])
        ax[i].set_ylabel("Cross-validated accuracy")
        ax[i].set_xlabel(p_name[i])
        ax[i].fill_between(p_j[i], p_i[i] + std_error, p_i[i] - std_error)

       
        ax[i].axhline(np.max(p_i[i]), linestyle='--', alpha=0.2)
        ax[i].set_xlim([p_j[i][0], p_j[i][-1]])
        if (i+1)%4==0:
            s+=1
            f=0
    plt.show()


#standard_deviation_knn()
#model_implementation()
#We get the best C = 0.2


#if we change parameter class_weight='balanced' of LOGIT, we drop accuracy to 82.6 from 88.8, best one is Default=None

#plt.show()
def get_mse_rmse_model(y_pred,state='',name=''):
    print("MSE "+name+" "+state+" : %.2f" % (metrics.mean_squared_error(y_val,y_pred)))
    print("MAE "+name+" "+state+" : %.2f" % (metrics.mean_absolute_error(y_val,y_pred)))
    print("RMSE "+name+" "+state+" : %.2f" % (np.sqrt(metrics.mean_squared_error(y_val,y_pred))))

def new_models():
    global logit2
    print "**********************************************"
    print "Neighbors = 27 is for best model KNeighborsClassifier"
    knn2= KNeighborsClassifier(n_neighbors=27)#Best neigbor
    knn2.fit(X_train,y_train)
    knnpred2=[]
    print "C=0.2 is best model for Logistic Regression for "
    logit2 = LogisticRegression(C=0.2)#Best Parameter
    logit2.fit(X_train,y_train)
    logitpred2=[]
    #DT
    print "max_depth=4 is best model for DT "
    
    d_tree1 = DecisionTreeClassifier(max_depth=4)#Best Parameter
    d_tree1.fit(X_train,y_train)
    dtreepred=[]
    #SVM
    print "Best Feature Selection - SVM 1.5"

    s_v_m1 = SVC(C=1.5)#Best Parameter
    s_v_m1.fit(X_train,y_train)
    s_v_pred=[]
            
    #ANN
    print "Best Feature Selection - ANN 0.071"
    
    a_n_n1 = MLPClassifier(alpha=0.071)#Best Parameter
    a_n_n1.fit(X_train,y_train)
    a_n_npred=[]
    
    for i in range(0,len(X_val)):
        knnpred2.append(knn2.predict([X_val[i]]))
        logitpred2.append(logit2.predict([X_val[i]]))
        dtreepred.append(d_tree1.predict([X_val[i]]))
        s_v_pred.append(s_v_m1.predict([X_val[i]]))
        a_n_npred.append(a_n_n1.predict([X_val[i]]))
    print "accuracy Of New KNN:",accuracy_score(y_val, knnpred2)
    print "accuracy Of New LogisticRegression:",accuracy_score(y_val, logitpred2)
    print "accuracy Of New Decision Tree:",accuracy_score(y_val, dtreepred)
    print "accuracy Of New SVM:",accuracy_score(y_val, s_v_pred)
    print "accuracy Of New ANN:",accuracy_score(y_val, a_n_npred)
    
    
    print "\n********************LOGISTIC*********************"
    
    print "New Model VS OLD Model For Logit"
    
    print('Logit Variance OLD: %.2f' % logit.score(X_val, y_val))# LOGISTIC 1 with Default MODEL
    print('Logit Variance NEW: %.2f' % logit2.score(X_val, y_val))# LOGISTIC 2 with best MODEL
     
    y_pred=logit.predict(X_val)# LOGISTIC 1 with Default Model
    get_mse_rmse_model(y_pred,'OLD','LOGIT')
    y_pred=logit2.predict(X_val)# LOGISTIC 2 with best MODEL
    get_mse_rmse_model(y_pred,'NEW','LOGIT')
    
    print "\n***************************KNN***********************"
    
    print "New Model VS OLD Model For Knn"
    
    print('KNN Variance OLD: %.2f' % knn.score(X_val, y_val))# KNN 1 with Default MODEL
    print('KNN Variance NEW: %.2f' % knn2.score(X_val, y_val))# KNN 2 with best MODEL
     
    y_pred=knn.predict(X_val)# KNN 3 with Default Model
    get_mse_rmse_model(y_pred,'OLD','KNN')
    
    y_pred=knn2.predict(X_val)# LOGISTIC 27 with best MODEL
    get_mse_rmse_model(y_pred,'NEW','KNN')
    print "*******************************************************"

    
    #####################################################################################
    print "New Model VS OLD Model For DT"
    
    print('DT Variance OLD: %.2f' % model.score(X_val, y_val))# DT 1 with Default MODEL
    print('DT Variance NEW: %.2f' % d_tree1.score(X_val, y_val))# DT 2 with best MODEL
     
    y_pred=model.predict(X_val)# DEfault
    get_mse_rmse_model(y_pred,'OLD','DT')
    
    y_pred=d_tree1.predict(X_val)# 
    get_mse_rmse_model(y_pred,'NEW','DT')
    print "*******************************************************"

    print "New Model VS OLD Model For SVM"
    
    print('SVM Variance OLD: %.2f' % svc.score(X_val, y_val))# SVC 1 with Default MODEL
    print('SVM Variance NEW: %.2f' % s_v_m1.score(X_val, y_val))# SVC 2 with best MODEL
     
    y_pred=model.predict(X_val)# OLD Default
    get_mse_rmse_model(y_pred,'OLD','SVM')
    
    y_pred=d_tree1.predict(X_val)# NEW
    get_mse_rmse_model(y_pred,'NEW','SVM')
    
    print "*******************************************************"

    print "New Model VS OLD Model For ANN"
    
    print('ANN Variance OLD: %.2f' % ann.score(X_val, y_val))# SVC 1 with Default MODEL
    print('ANN Variance NEW: %.2f' % a_n_n1.score(X_val, y_val))# SVC 2 with best MODEL
     
    y_pred=model.predict(X_val)# OLD Default
    get_mse_rmse_model(y_pred,'OLD','ANN')
    
    y_pred=d_tree1.predict(X_val)# NEW
    get_mse_rmse_model(y_pred,'NEW','ANN')

    #TEST
    print "********************TEST best parameters**************************"
    knnpred_test=[]
    logitpred_test=[]
    svm_test=[]
    ann_test=[]
    dt_test=[]
    for i in range(0,len(X_test)):
        knnpred_test.append(knn2.predict([X_test[i]]))
        logitpred_test.append(logit2.predict([X_test[i]]))
        svm_test.append(s_v_m1.predict([X_test[i]]))
        ann_test.append(a_n_n1.predict([X_test[i]]))
        dt_test.append(d_tree1.predict([X_test[i]]))
    print "accuracy knn TEST:",accuracy_score(y_test, knnpred_test)
    print "accuracy logistic TEST:",accuracy_score(y_test, logitpred_test)
    print "accuracy SVM TEST:",accuracy_score(y_test, svm_test)
    print "accuracy DT TEST:",accuracy_score(y_test, dt_test)
    print "accuracy ANN TEST:",accuracy_score(y_test, ann_test)
    

#Checking Accuracy and ERRORs
''' Confusion Matrix: It is nothing but a tabular representation of Actual vs Predicted values. 
This helps us to find the accuracy of the model and avoid overfitting. This is how it looks like:'''
'''
Logistic Regression Analysis - linear regressions deal with continuous valued series whereas a logistic regression deals with categorical (discrete) values. Discrete values are difficult to work with because they are non differentiable so gradient-based optimization techniques don't apply.
'''
''' Feature selection
    Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
    Improves Accuracy: Less misleading data means modeling accuracy improves.
    Reduces Training Time: Less data means that algorithms train faster.'''
def Tree_class():
    # fit an Extra Trees model to the data
    # display the relative importance of each attribute
    model_Tree = ExtraTreesClassifier()
    model_Tree.fit(X_train,y_train)
    print (model_Tree.feature_importances_)
    
def get_mse_rmse(y_val_new,y_pred):
    print("MSE3: %.2f" % (metrics.mean_squared_error(y_val_new,y_pred)))
    print("MAE3: %.2f" % (metrics.mean_absolute_error(y_val_new,y_pred)))
    print("RMSE3: %.2f" % (np.sqrt(metrics.mean_squared_error(y_val_new,y_pred))))
def accuracy_metrics_for_selected_features():
    global logit2
    #So,Looking at Important features, we Selected them, 
    #and implementation time decreased,also RMSE AND MSE.
    xx=frame[['Income','Residence_Length']]
    yy=frame['Buy']
    xx= list(np.array(xx))
    yy=list(np.array(yy))
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(xx, yy, test_size=0.2, random_state=3)#20% Test, 80%Train
    X_train_new,X_val_new, y_train_new, y_val_new = train_test_split(X_train,y_train, test_size=0.25,random_state=3)#20% Validation 60%Train
    
    #select best Parameter for the best Features
    #logit-3
    print "Best Feature Selection - Logistic Regression"
    logit3 = LogisticRegression(C=0.2)#Best Parameter
    logit3.fit(X_train_new,y_train_new)
    logitpred3=[]
    for i in range(0,len(X_val_new)):
        logitpred3.append(logit3.predict([X_val_new[i]]))
    print "accuracy Of New LogisticRegression:",accuracy_score(y_val_new, logitpred3)
    y_pred=logit3.predict(X_val_new)# LOGISTIC 3 with best MODEL
    get_mse_rmse(y_val_new,y_pred)
    
    #####################################################################################
    print "\nBest Feature Selection - Decision Tree"
    d_tree = DecisionTreeClassifier(max_depth=4)#Best Parameter
    d_tree.fit(X_train_new,y_train_new)
    dtreepred=[]
    for i in range(0,len(X_val_new)):
        dtreepred.append(d_tree.predict([X_val_new[i]]))
    
    print "accuracy Of New Decision Tree:",accuracy_score(y_val_new, dtreepred)
    y_pred=d_tree.predict(X_val_new)# LOGISTIC 3 with best MODEL
    get_mse_rmse(y_val_new,y_pred)
    #####################################################################################
    print "\nBest Feature Selection - KNN "
    k_nn = KNeighborsClassifier(n_neighbors=27)#Best Parameter
    k_nn.fit(X_train_new,y_train_new)
    k_nnpred=[]
    for i in range(0,len(X_val_new)):
        k_nnpred.append(k_nn.predict([X_val_new[i]]))
    
    print "accuracy Of New KNN:",accuracy_score(y_val_new, k_nnpred)
    y_pred=k_nn.predict(X_val_new)# LOGISTIC 3 with best MODEL
    get_mse_rmse(y_val_new,y_pred)


    #####################################################################################
    print "Best Feature Selection - SVM"

    s_v_m = SVC(C=1.5)#Best Parameter
    s_v_m.fit(X_train_new,y_train_new)
    s_v_pred=[]
    for i in range(0,len(X_val_new)):
        s_v_pred.append(s_v_m.predict([X_val_new[i]]))
    

    print "accuracy Of New SVM:",accuracy_score(y_val_new, s_v_pred)
    y_pred=s_v_m.predict(X_val_new)# LOGISTIC 3 with best MODEL
    get_mse_rmse(y_val_new,y_pred)
    #####################################################################################
    print "Best Feature Selection - ANN"
    
    a_n_n = MLPClassifier(alpha=0.071)#Best Parameter
    a_n_n.fit(X_train_new,y_train_new)
    a_n_npred=[]
    for i in range(0,len(X_val_new)):
        a_n_npred.append(a_n_n.predict([X_val_new[i]]))
    
    print "accuracy Of New ANN:",accuracy_score(y_val_new, a_n_npred)
    y_pred=a_n_n.predict(X_val_new)# LOGISTIC 3 with best MODEL
    get_mse_rmse(y_val_new,y_pred)

#accuracy_metrics_for_selected_features()
#new_models()
    
#Using RandomForestRegressor, we will select features with high values, which mean those features are important
#but, it takes huge amount of Time to implement 17 features.
'''
Univariate feature selection is in general best to get a better understanding of the data,
its structure and characteristics. It can work for selecting top features for model
improvement in some settings, but since it is unable to remove redundancy 
(for example selecting only the best feature among a subset of strongly correlated features)
'''
'''
A random forest regressor.
A random forest is a meta estimator that fits a number of classifying decision trees on various
sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
'''
def feature_importance_random_forest():
    names = features[1:]
    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = []
    for i in range(X.shape[1]):
         score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                                  cv=ShuffleSplit(len(X), 3, .3))
         scores.append((round(np.mean(score), 3), names[i]))
    print sorted(scores, reverse=True)





inp=''
while inp!='x':
    print "1 - Correlations"
    print "2 - Visualize correlation figure "
    print '3 - Visualize scatter_matrix figure'
    print '4 - Visualize only highly correlated features'
    print '5 - Visualize histogram figure'
    print '6 - Print General accuracy for all appropriate algorithms'
    print '7 - Visualize Model implementation'
    print "8 - Show newly generated Model's performation and accuracy"
    print "9 - Confusion Matrix and for knn and statistics"
    
    print "10 - Confusion Matrix and for logit and statistics"
    print "11 - Get feature Importance using ExtraTreeClassifier"
    print "12 - New_Model from Selecting important features, and their accuracy,errors,etc"
    print "13 - Get feature Importance using RandomForestClassifier"
    #print "14 - Confusion Matrix and for logit and statistics old model"
    
    
    
    print 'x - To exit'
    
    inp=raw_input('Enter The command: ')
    if inp=='1':
        Thread(target=get_correlations).start()   
    elif inp=='2':
        correlation_fig()
    elif inp=='3': 
        scatter_matrix_fig()
    elif inp=='4':
        draw_high_cor()
    elif inp=='5':
        hist_fig()
    elif inp=='6':
        Thread(target=general_accuracy).start()
    elif inp=='7':
        model_implementation()
    elif inp=='8':
        Thread(target=new_models).start()
    elif inp=='9':
        confusion_matrix_class_k()
    elif inp=='10':
        confusion_matrix_class_l()
    elif inp=='11':
        Tree_class()
    elif inp=='12':
        accuracy_metrics_for_selected_features()
    elif inp=='13':
        print 'You have to wait until it performes... about 3-5minutes...'
        feature_importance_random_forest()

    elif inp=='x':
        print 'Exiting...'
    else:
        print 'No such command'
    time.sleep(2)
    

