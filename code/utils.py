# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 15:11:32 2017

@author: ravikiran
"""
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn import datasets
import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,\
                              BaggingClassifier, RandomForestRegressor,\
                              BaggingRegressor, ExtraTreesRegressor
le = preprocessing.LabelEncoder()

def get_forest_size(forest):
    num_leaves = 0
    for estimator in forest:
        num_leaves += estimator.tree_.node_count
    return num_leaves
def get_models(model_name, predicttype):
    """ set of models to be tested """
    models_dict = {}
    if(predicttype=='classify'):
        models_dict['RandomSubspace'] = RandomForestClassifier(bootstrap=False)
        models_dict['RandomForest'] = RandomForestClassifier(bootstrap=True)
        models_dict['Bagger'] = BaggingClassifier()
        models_dict['ExtraTrees'] = ExtraTreesClassifier(bootstrap=True)
    else:
        models_dict['RandomSubspace'] = RandomForestRegressor(bootstrap=False)
        models_dict['RandomForest'] = RandomForestRegressor(bootstrap=True)
        models_dict['Bagger'] = BaggingRegressor()
        models_dict['ExtraTrees'] = ExtraTreesRegressor(bootstrap=True)
    return models_dict[model_name]

def helper_parallel(args):
    """wrapper to dispatch any function with its args in parallel"""
    eval_func = args[0]
    eval_func_args = args[1:]
    return eval_func(*eval_func_args)

def plot_scores(models, plot_vals, n_trees, dataset, file_name):
    """ final size and accuracy scores plot function """                
    size_mat, acc_mat, alpha_vals = plot_vals
    
    plt.figure()
    plt.xticks(range(len(models)), models)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].boxplot(size_mat.T, labels=models, showmeans=True, meanline=True)
    axes[0].set_ylabel('Size Ratio')
    axes[0].set_ylim([0, 1])
    axes[1].boxplot(acc_mat.T, labels=models, showmeans=True, meanline=True)
    axes[1].set_ylabel('Acc. Ratio')
    axes[1].set_ylim([0, 2])
    fig.suptitle('Size rations and accuracies #trees = '
                +repr(n_trees) + ', Dataset : ' + dataset)
    plt.savefig(file_name+'.png')
    plt.close('all')
    
    plt.figure()
    plt.plot(alpha_vals[0], label=r'$\alpha^{\ast}$'+'-Tree')
    plt.plot(alpha_vals[1], label=r'$\alpha^{\ast}$'+'-Forest')
    plt.title('Averaged Optimal CC parameters')
    plt.xlabel('Tree index')
    plt.ylabel('CC parameter :'+  r'$\alpha^{\ast}$')
    plt.legend(loc='best')
    plt.savefig(file_name+'_alphas_'+'.png')
    
    return

def save_results(save_args):
    """ Plots ratios and writes txt with test accuracies on pruned and
    unpruned forests
    """
    (opt_alpha, prune_ratio) = save_args['subplots']
    (model_name, dataset, n_trees) = save_args['names']
    (acc_tr_forest, acc_test_forest, acc_test_pruned) = save_args['accuracies']
    result_path = save_args['result_path']

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(opt_alpha, label='Alpha')
    axarr[0].legend(loc='best')
    axarr[0].set_title('Optimum alphas')
    axarr[1].plot(prune_ratio, label='PruneRatio')
    axarr[1].legend(loc='best')
    axarr[1].set_title('#Leaves Ratio Pruned-to-unpruned')
    axarr[1].set_xlabel('Trees j')
    axarr[0].set_ylabel(r'$\alpha^{\ast}_j$')
    axarr[1].set_ylabel('#Leaves')
    print('')
    file_name = model_name+'-'+dataset+'t'+repr(n_trees)+ 'alpha_leaves' + '.png'
    f.savefig(result_path+file_name)

    file_name = model_name+'-'+dataset+'t'+repr(n_trees)+ "Accuracies.txt"
    with open(result_path+file_name, "w") as text_file:
        print('-> Orig  acc X_test  = ' +repr(acc_test_forest), file=text_file)
        print('-> Prune acc X_test  = ' +repr(acc_test_pruned), file=text_file)

    plt.close('all')
    return

def drawProgressBar(percent, barLen = 20):
    """ Function simply draws a progress bar """

    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

def get_breast_cancer_dataset():

#    url="http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
#    s=requests.get(url).content
    le = preprocessing.LabelEncoder()
    cols = []
    cols.append("Sample code number")
    cols.append("Clump Thickness")
    cols.append("Uniformity of Cell Size")
    cols.append("Uniformity of Cell Shape")
    cols.append("Marginal Adhesion")
    cols.append("Single Epithelial Cell Size")
    cols.append("Bare Nuclei")
    cols.append("Bland Chromatin")
    cols.append("Normal Nucleoli")
    cols.append("Mitoses")
    cols.append("Class")
    df=pd.read_csv('./mldata/mldata/breast-cancer-wisconsin.data', names=cols)
    df['Bare Nuclei'] = le.fit_transform(df['Bare Nuclei'])
    newdf = df[df.columns[1:11]]
    y = newdf['Class'].values
    X = newdf.ix[:,1:10].as_matrix()
    return X, y #returns without ID

def get_digits():
    digits = datasets.load_digits()
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    return X, y

def get_iris():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    return X, y

def get_mnist():
    mnist = pickle.load( open( "./mldata/mnist.p", "rb" ) )
    X = mnist.data
    y = le.fit_transform(mnist.target)
    return X, y

def get_breast_cancer():
    data = datasets.load_breast_cancer()
    return data.data, data.target

def mouse_protein_data():
    df = pd.read_excel('./mldata/Mice-protein/Data_Cortex_Nuclear.xls')
    y = le.fit_transform(df['class'].values)
    df['MouseID'] = le.fit_transform(df['MouseID'])
    df['Genotype'] = le.fit_transform(df['Genotype'])
    df['Treatment'] = le.fit_transform(df['Treatment'])
    df['Behavior'] = le.fit_transform(df['Behavior'])
    df = df.fillna('both')
    X = df.ix[:,1:81].as_matrix()
    return X, y

def get_forest_coverage():
    train = pd.read_csv('./mldata/forests_coverage/train.csv')
#    test = pd.read_csv('./mldata/forests_coverage/test.csv')

    keys = list(train.keys())[1:-1]
    y_train = le.fit_transform(train['Cover_Type'])
    X_train = np.zeros(shape=(train.shape[0], len(keys)), dtype=int)
    for i, key in enumerate(keys):
        X_train[:,i] = le.fit_transform(train[key])
#    y_test = le.fit_transform(test['Cover_Type'])
#    X_test = np.zeros(shape=(test.shape[0], len(keys)), dtype=int)
#    for i, key in enumerate(keys):
#        X_test[:,i] = le.fit_transform(test[key])
    return X_train, y_train

def get_breast_tissue():
    df = pd.read_excel('./mldata/breast_tissue/Data.xlsx')
    y = le.fit_transform(df['Class'])
    X = df.ix[:,2::].as_matrix()
    return X, y

def get_wine_quality(color):
    if(color=='red'):
        df = pd.read_csv('./mldata/wine_quality/winequality-red.csv',sep=';')
    else:
        df = pd.read_csv('./mldata/wine_quality/winequality-white.csv',sep=';')
    X = df.ix[:,0:11].as_matrix()
    y = le.fit_transform(df['quality'].as_matrix())
    return X, y

def get_leaf_data():
    df = pd.read_csv('./mldata/leaf/leaf_mod.csv')
    y = df['Class'].values
    X = df.ix[:,2:].as_matrix()
    return X, y

def get_abalone():
    df = pd.read_csv('mldata/abalone.csv')
    df['Sex'] = le.fit_transform(df['Sex'])
    X = df.ix[:,1:10].as_matrix()
    r = df['Rings']
    y = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        if(1<=r[i]<=8):
            y[i] = 0
        elif(9<=r[i]<=10):
            y[i] = 1
        else:
            y[i] = 2
    return X, y

def get_boston():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    return X, y

def get_diabetes():
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return X, y

def get_linnerud():
    linnerud = datasets.load_linnerud()
    X = linnerud.data
    y = linnerud.target
    return X, y[:,2]

def get_split_sizes(name, predicttype):
    if(predicttype=='classify'):
        test_size={'abalone' : 0.1,
                   'digits' : 0.1,
                   'iris': 0.25,
                   'redwine': 0.1,
                   'breast_cancer': 0.1,
                   'whitewine' : 0.5,
                   'forest_coverage' : 0.5,
                   'mnist' : 0.5}
    else:
        test_size={'boston' : 0.2,
           'diabetes' : 0.2,
           'linnerud' : 0.2}
    return test_size[name]

def get_dataset(name, predicttype):
    if(predicttype=='classify'):
        dataset_functions = {
        "digits": get_digits(),
        "mnist" : get_mnist(),
        "iris" : get_iris(),
        "breast_cancer" : get_breast_cancer(),
        "leaf" : get_leaf_data(),
        "forest_coverage" : get_forest_coverage(),
        "breast_tissue" : get_breast_tissue(),
        "redwine" : get_wine_quality("red"),
        "whitewine": get_wine_quality("white"),
        "abalone" : get_abalone(),
        "blob" : datasets.make_classification(n_samples=1000)}
    else:
        dataset_functions = {
        "boston" : get_boston(),
        "diabetes" : get_diabetes(),
        "linnerud" : get_linnerud()}

    return dataset_functions[name]

def plot_accuracy(acc_list, acc_ensemble, plot_params):
    (dataset, model, CV_set, n_trees) = plot_params
    plt.plot(acc_list, label='pruned')
    plt.plot(len(acc_list)*[acc_ensemble],label=model)
    plt.title('Ensemble Vs Pruned (' + repr(n_trees) +' trees)' +\
              ': Model =' + model + ', Dataset = ' + dataset +\
              ' CV set =' + CV_set)
    plt.xlabel('Number of prunes')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    return
    
def _plot_xval_error(alpha_list, acc_list_train, acc_list_test):
    """ Plot train and test errors for decision tree over n-folds of data """
    plt.figure()
    plt.gca().invert_xaxis()
    plt.plot(alpha_list, acc_list_test, 'r', label='Test')
    plt.plot(alpha_list, acc_list_train,'b', label='Train')
    plt.title('Classifcation accuracy', {'color': 'b','fontsize': 20})
    plt.xlabel('Cost-Complexity Parameter(' + r'$\alpha$' + ')' ,
                {'color': 'b','fontsize': 20})
    plt.ylabel('Accuracy',{'color': 'b','fontsize': 20})
#        plt.text(1.01, -0.02, "-1", {'color': 'k', 'fontsize': 20})
    plt.legend(loc='best')