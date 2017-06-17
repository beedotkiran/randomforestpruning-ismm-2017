# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:06:55 2017

@author: ravikiran
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import _generate_unsampled_indices as get_unsampled_indices
from sklearn.ensemble.forest import _generate_sample_indices as get_insample_indices

from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed
from copy import deepcopy as dcopy

from utils import *
from prune_funcs import *
import os
import sys

def simple_prune_demo():
    """ Example of CC pruning : example with internal nodes"""
    n_samples=25
    n_classes=3

    X, y = make_blobs(n_samples=n_samples, centers=n_classes,cluster_std=2.0)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.01)
    estimator = DecisionTreeClassifier()
    estimator.fit(X_train, y_train)

    c_l = estimator.tree_.children_left
    c_r = estimator.tree_.children_right

    #substitution error scores
    r = get_error_function(estimator)
    p = estimator.tree_.n_node_samples/X_train.shape[0]
    print(c_l)
    print(c_r)
    print('----------------------------------')
    #get leaves
    leaves, i_nodes = get_leaves_at_node(0, c_l, c_r)
    print(leaves, i_nodes)
    alpha = calculate_alpha(r, p, i_nodes, c_l, c_r)
    min_alpha_node = min(alpha, key=alpha.get)

    ##prune 1
    print('Pruning leaves under node '+repr(min_alpha_node))
    c_l, c_r = prune_list(min_alpha_node, c_l, c_r)
    leaves, i_nodes = get_leaves_at_node(0,c_l,c_r)
    print(leaves, i_nodes)

    #prune 2
    alpha = calculate_alpha(r, p, i_nodes, c_l, c_r)
    if(alpha):
        min_alpha_node = min(alpha, key=alpha.get)
        print('Pruning leaves under node '+repr(min_alpha_node))
        c_l, c_r = prune_list(min_alpha_node, c_l, c_r)
        leaves, i_nodes = get_leaves_at_node(0,c_l,c_r)
        print(leaves, i_nodes)

    #testing node_list
    c_l = estimator.tree_.children_left
    c_r = estimator.tree_.children_right
    tree_leaves_id = estimator.apply(X_train)
    leaves, i_nodes = get_leaves_at_node(0,c_l,c_r)
    alpha_list = []
    node_list = []
    leaves_id_list = {}
    leaves_id_list[0] = dcopy(tree_leaves_id)
    print('Pruning sequences : ')
    while(len(i_nodes)):
        print(i_nodes)
        print(alpha_list)
        print(tree_leaves_id)
        print('-----------------')
        alpha = calculate_alpha(r, p, i_nodes, c_l, c_r)
        if(alpha):
            min_alpha_node = min(alpha, key=alpha.get)
            min_alpha = alpha[min_alpha_node]
            alpha_list.append(min_alpha)
            node_list.append(min_alpha_node)
            c_l_, c_r_ = prune_list(min_alpha_node, c_l, c_r)
            leaves_, i_nodes = get_leaves_at_node(0,c_l_,c_r_)
            #relabel test labels to new leaves
            for l in leaves_:
                old_l, _ = get_leaves_at_node(l,c_l,c_r)
                for leaf in old_l:
                    tree_leaves_id[tree_leaves_id==leaf] = l
            leaves_id_list[min_alpha_node] = dcopy(tree_leaves_id)
            #update new parameters
            c_l, c_r = c_l_, c_r_
    print('Node list')
    print(node_list)
    print('Alpha list')
    print(alpha_list)
    print('Leaves IDs')
    print(leaves_id_list)
    return

def cc_pruning_tree(estimator, X_train, X_test, y_train, y_test, 
                                plot_flag=True):
    """ Implements cost complexity pruning
    Inputs
    ------
    estimator : decision tree base classifier
    X_train, y_train : Training set
    X_test, y_test : Testing set

    Outputs
    -------
    num_leaves,    : number of leaves at each pruning step
    acc_list_test, : Accuracy over test set at each pruning step
    acc_list_train,: Accuracy over training at each pruning step
    alpha_list     : Values of alpha (CC parameter) that provides ordering
                    of nodes to achieve each pruning step
    """

    n_nodes = estimator.tree_.node_count
    c_l = estimator.tree_.children_left
    c_r = estimator.tree_.children_right

    #substitution error scores
    r = get_error_function(estimator)
    p = estimator.tree_.n_node_samples/X_train.shape[0]
    #prune and predict
    node_indicator = estimator.decision_path(X_train)
    train_leaves_id = estimator.apply(X_train) #change here for train

    num_leaves = []
    acc_list_train = []
    alpha_list = []
    node_list = []

    leaves, internal_nodes = get_leaves_at_node(0, c_l, c_r)
    node_score= get_class_prob(estimator)
    while(len(internal_nodes)):
        num_leaves.append(len(np.unique(train_leaves_id)))
        y_predicted = predict_tree(node_score, train_leaves_id, 'classify')
        acc_list_train.append(accuracy_score(y_train, y_predicted))
        alpha = calculate_alpha(r, p, internal_nodes, c_l, c_r)
        if(alpha):
            min_alpha_node = min(alpha, key=alpha.get)
            alpha_list.append(alpha[min_alpha_node])
            node_list.append(min_alpha_node)
            c_l_, c_r_ = prune_list(min_alpha_node, c_l, c_r)
            leaves_, internal_nodes = get_leaves_at_node(0,c_l_,c_r_)
            #relabel test labels to new leaves
            for l in leaves_:
                old_l, _ = get_leaves_at_node(l,c_l,c_r)
                for leaf in old_l:
                    train_leaves_id[train_leaves_id==leaf] = l
            #update new parameters
            c_l, c_r = c_l_, c_r_
        else:
            print('Done with pruning')
            break

    #predict on test set with pruned trees
    acc_list_test = []
    c_l = estimator.tree_.children_left
    c_r = estimator.tree_.children_right
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    leaves, _ = get_leaves_at_node(0, c_l, c_r)
    is_leaves[leaves] = True
    test_leaves_id = estimator.apply(X_test) #change here for train
#    acc_list_test.append(0)
    for node in node_list:
        y_predicted = predict_tree(node_score, test_leaves_id, 'classify')
        acc_list_test.append(accuracy_score(y_test, y_predicted))
        c_l_, c_r_ = prune_list(node, c_l, c_r)
        leaves_, _ = get_leaves_at_node(0,c_l_,c_r_)
        #relabel test labels to new leaves
        for l in leaves_:
            old_l, _ = get_leaves_at_node(l,c_l,c_r)
            for leaf in old_l:
                test_leaves_id[test_leaves_id==leaf] = l
        #update new parameters
        c_l, c_r, leaves = c_l_, c_r_, leaves_
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        is_leaves[leaves] = True
    opt_idx = np.argmax(acc_list_test)
#    alpha_list = [min(alpha_list)] + alpha_list#prepended smallest alpha value
#    print('Optimal alpha ='+repr(alpha_list[opt_idx]))
#    print(opt_idx)
    print('#Leaves = ' + repr(num_leaves))
    def plot_xval_error_bound():
        upper_diff = np.mean(acc_list_test)+0.95*np.std(acc_list_test)
        lower_diff = np.mean(acc_list_test)-0.95*np.std(acc_list_test)
        upper_vec = np.ones(len(acc_list_test))*upper_diff
        lower_vec = np.ones(len(acc_list_test))*lower_diff
        plt.gca().invert_xaxis()
        plt.plot(alpha_list, acc_list_test, 'r', label='Test')
        plt.plot(alpha_list, acc_list_train,'b', label='Train')
        plt.plot(alpha_list, upper_vec, 'g', label='1-SE')
        plt.plot(alpha_list, lower_vec, 'g')
        plt.title('Pruning Accuracy', {'color': 'b','fontsize': 20})
        plt.xlabel('Cost-Complexity Parameter(' + r'$\alpha$' + ')' ,
                    {'color': 'b','fontsize': 20})
        plt.ylabel('Accuracy',{'color': 'b','fontsize': 20})
#        plt.text(1.01, -0.02, "-1", {'color': 'k', 'fontsize': 20})
        plt.legend(loc='best')

    def plot_xval_error():
        plt.figure(1)
        opt_alpha_vec = np.zeros(len(acc_list_test))
        opt_alpha_vec[opt_idx] = 1
        plt.gca().invert_xaxis()
        plt.plot(alpha_list, acc_list_test, 'r', label='Test')
        plt.plot(alpha_list, acc_list_train,'b', label='Train')
        
        plt.title('Classifcation accuracy', {'color': 'b','fontsize': 20})
        plt.xlabel('Cost-Complexity Parameter(' + r'$\alpha$' + ')' ,
                    {'color': 'b','fontsize': 20})
        plt.ylabel('Accuracy',{'color': 'b','fontsize': 20})
#        plt.text(1.01, -0.02, "-1", {'color': 'k', 'fontsize': 20})
        plt.legend(loc='best')
        plt.figure(2)
        plt.plot(alpha_list, num_leaves, label='#Leaves')
        plt.title('Cost-Complexity Parameter Vs #Leaves')
        plt.xlabel('Cost-Complexity Parameter(' + r'$\alpha$' + ')' ,
                    {'color': 'b','fontsize': 20})
        plt.ylabel('#Leaves',{'color': 'b','fontsize': 20})
#        ltext = plt.gca().get_legend().get_texts()
#        for l in ltext:
#            plt.setp(l, fontsize=16)
#        plt.show()
#    plt.figure()

    if(plot_flag):
        plot_xval_error()
    return num_leaves, acc_list_test, acc_list_train, alpha_list

def test_plot_CCpruning_parameters(n_samples = 500, num_classes = 5):

    X, y = make_blobs(n_samples=n_samples, centers=num_classes, cluster_std=2.0)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
    estimator = DecisionTreeClassifier()
    estimator.fit(X_train, y_train)
    alpha_list = cc_pruning_tree(estimator, X_train, X_test,
                                         y_train, y_test)
#    plt.figure()
#    plt.hist(alpha_list)
#    plt.title('Distriubution of Cost-complexity parameter')
#    plt.xlabel('$alpha$s')
#    plt.ylabel('Frequency')
    return alpha_list

def test_alpha_distribution(n_samples = 500, num_classes = 4, n_trees = 5, 
                            test_size = 0.1):
    """ Function calculates cost-complexity parameters (alpha) values 
    across different trees, decision tree classifier ensembles.
    Inputs 
    ------
    n_samples : number of input samples
    num_classes : number of classes in classification sample dataset

        Random Forest Classifier (Bagged)
        Random Subspace Trees (unbagged RF)

        Observation : distrbution in case of bagging is more uniform
    """

    X, y = make_classification(n_samples=n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, 
                                                        random_state=0)
    forest = ExtraTreesClassifier(n_estimators=n_trees)
    forest.fit(X_train, y_train)

    n_bins = 20

    plt.figure()
    for t, estimator in enumerate(forest):
        print('->Tree '+repr(t))
        alpha_list = cc_pruning_tree(estimator, X_train, X_test,
                                                  y_train, y_test)
        plt.hist(alpha_list, bins=n_bins, alpha = 0.65, label='tree#'+repr(t))
        plt.title('Cost-complexity parameter in random subspace ensemble with'
                  +  repr(n_trees) + ' trees')
        plt.xlabel('$alpha$s')
        plt.ylabel('Frequency')
    plt.legend(loc='best')

    forest = RandomForestClassifier(bootstrap=False, n_estimators=n_trees)
    forest.fit(X_train, y_train)
    plt.figure()
    for t, estimator in enumerate(forest):
        print('->Tree '+repr(t))
        alpha_list = cc_pruning_tree(estimator, X_train, X_test,
                                                  y_train, y_test)
        plt.hist(alpha_list, bins=n_bins, alpha = 0.65, label='tree#'+repr(t))
        plt.title('Cost-complexity parameter in random forest ensemble with'
                  +  repr(n_trees) + ' trees')
        plt.xlabel('$alpha$s')
        plt.ylabel('Frequency')
    plt.legend(loc='best')
    return

def plot_ISMM_sub(min_estimators = 15, max_estimators = 200):
    """ plots for ISMM (2) OOB error """
    data = _get_data('iris', 0.1, 'classify') 
    X_train, X_test, y_train, y_test = data
    model = RandomForestClassifier(bootstrap=True, oob_score=True)

    # Range of `n_estimators` values to explore.
    oob_error = []
    for i in range(min_estimators, max_estimators + 1):
        print(repr(i) + ' trees')
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
        oob_error.append((i,1 - model.oob_score_))
    plt.figure()
    plt.plot()
    plt.xlim(min_estimators, max_estimators)
    xs, ys = zip(*oob_error)
    plt.plot(xs,ys)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")

    """ getting the indicies that are out-of-bag for each tree """
    oob_error = 1 - model.oob_score_
    print('OOB ERROR size : ')
    print(oob_error.shape)
    # X here - training set of examples
    def test_oob_samples():
        n_samples = X.shape[0]
        unsampled_indices = {}
        sample_indices = {}
        for i, tree in enumerate(model.estimators_):
            # Here at each iteration we obtain out of bag samples for each tree.
            rand_state = tree.random_state
            sample_indices[i] = get_insample_indices(rand_state, n_samples)
            unsampled_indices[i] = get_unsampled_indices(rand_state,
                                                                n_samples)
        if(0):
            i = 1
            X_tr = list(sample_indices[i])
            X_oob = list(unsampled_indices[i])
            flag = np.all(set(X_tr).union(X_oob) == set(range(n_samples)))
            if(flag):
                print('OOB + in bag samples = training set!!!')
    return

def plot_tree_CV_error(dataset = 'redwine', n_fold = 1, test_size = 0.2):
    """ Plots for ISMM (1) : CV error for CC pruning using decision tree """

    alpha_list = {}
    plot_flag = True
    if(plot_flag):
        plt.figure()
        plt.gca().invert_xaxis()

    max_acc = 0
    for f in range(n_fold):
        data = _get_data(dataset, test_size, 'classify')
        tree = DecisionTreeClassifier()
        X_train, X_test, y_train, y_test = data
        tree.fit(X_train, y_train)
        results_ = cc_pruning_tree(tree, X_train, X_test, y_train, y_test, 
                                    plot_flag=plot_flag)
        num_leaves, acc_list_test, acc_list_train, alpha_list = results_
        if(max_acc > max(acc_list_test) and f > 1):
            print('')
        else:
            max_acc = max(acc_list_test)
            opt_index = np.argmax(acc_list_test)
            opt_alpha = alpha_list[opt_index]

        print('Fold#'+repr(f) + 'accuracy = ' +repr(max_acc)+ 
             '-Alpha*='+repr(opt_alpha))

    if(debug_flag):
#        plt.figure()
#        plt.gca().invert_xaxis()
#        plt.plot(mean_test_leaves, mean_test_acc, 'r', label='mean-test-accuracy')
#        plt.plot(mean_train_leaves, mean_train_acc, 'b', label='mean-train-accuracy')
#        plt.legend(loc='best')
#        plt.title('Mean Prediction Error('+repr(n_fold)+' folds) Dataset : '\
#                  +  dataset, {'color': 'b','fontsize': 20})
#        plt.xlabel('Num Leaves',{'color': 'b','fontsize': 20})
#        plt.ylabel('Accuracy',{'color': 'b','fontsize': 20})
#    #    plt.text(1.01, -0.02, "-1", {'color': 'k', 'fontsize': 20})
#        plt.legend(loc='best')
#        ltext = plt.gca().get_legend().get_texts()
#        for l in ltext:
#            plt.setp(l, fontsize=16)
#        plt.show()
#        #alpha plots
        plt.figure()
        for fold in range(n_fold):
            plt.plot(alpha_list[fold], label='fold#'+repr(fold))
        plt.title('Cost-Complexity parameters', {'color': 'b','fontsize': 20})
        plt.xlabel('Index',{'color': 'b','fontsize': 20})
        plt.ylabel('Cost-complexity parameter ' + r'$\alpha$',{'color': 'b','fontsize': 20})
        plt.legend(loc='best')
        ltext = plt.gca().get_legend().get_texts()
        for l in ltext:
            plt.setp(l, fontsize=16)

    return opt_alpha

def test_prepruning(n_iter = 10, n_cores = 4, dataset = 'whitewine',
                    predicttype='classify', test_size=0.1,
                    model_name = 'RandomForest', n_trees = 100):
    """
    Test to evaluate the performance of pre-pruning of tree based classifiers.
    The two ways evaluate the CV errors are used to determine the value of the
    optimal pre-pruning paramter value.

    """

    X, y = get_dataset(dataset, predicttype)
    depth_list = range(1,100,10)
    acc_tr_forest, acc_test_forest = [],[]
    for i, d in enumerate(depth_list):
        acc_test_ = [0]*n_iter
        acc_tr_ = [0]*n_iter
        forest = get_models(model_name, predicttype)
        forest.set_params(n_estimators = n_trees, n_jobs=n_cores, max_depth=d)
        for n in range(n_iter):
            res_ = train_test_split(X,y,test_size=test_size)
            X_train, X_test, y_train, y_test = res_
            forest.fit(X_train, y_train)
            y_test_pred = forest.predict(X_test)
            y_train_pred = forest.predict(X_train)
            acc_test_[n] = get_acc_score(y_test, y_test_pred, predicttype)
            acc_tr_[n] = get_acc_score(y_train, y_train_pred, predicttype)
        acc_test_forest.append(np.mean(acc_test_))
        acc_tr_forest.append(np.mean(acc_tr_))
        print(repr((i,d))+' : '+repr((acc_tr_forest[i],acc_test_forest[i])))

    acc_forest2 = [0]*n_iter
    forest2 = get_models(model_name, predicttype)
    forest2.set_params(n_estimators=n_trees, n_jobs=n_cores)
    for n in range(n_iter):
        res_ = train_test_split(X,y,test_size=test_size)
        X_train, X_test, y_train, y_test = res_
        forest2.fit(X_train, y_train)
        y_test_pred = forest2.predict(X_test)
        acc_forest2[n] = get_acc_score(y_test, y_test_pred, predicttype)
    plt.figure()
    plt.plot(depth_list, acc_tr_forest, label='train')
    plt.plot(depth_list, acc_test_forest, label='test')
    plt.plot(depth_list, [np.mean(acc_forest2)]*len(depth_list), label='no-depth')
    plt.legend(loc='best')
    plt.title('Test error Vs Max. Depth :' + model_name+'-'+dataset)
    plt.xlabel('Maximum depth of Trees')
    plt.ylabel('Classification Accuracy')
    plt.show()

    return

def CCpruning_withCV(forest, data, CV_set='OOB-set', predicttype='classify'):
    """
    Calculate the \mathcal{A}_j, \mathcal{T}_j for each tree T_j
    IMPLEMENTED
        1. alpha*_j = Crossvalidate on OOB_j set, min. CV error per tree
            - better size ratio
            - Same or better accuracy on test, min. CV error per tree
        2. alpha*_j = Crossvalidate on X_Train, min. CV error per tree
            - same sizes and same accuracy on test
    TODO
        3. alpha*_j = Crossvalidate on X_Train, min. CV error on forest
        4. alpha*_j = Crossvalidate on OOB_j set, min. CV error on forest

    Inputs
    ------

    CV_set : 'OOB-set' or 'train-set'
    predicttype : 'classify'/'regress'
    forest : forest data structure from scikitlearn to prune
    data : tuple of (X_train, X_test, y_train, y_test)

    Outputs
    -------
    prune_ratio : ratio between size of pruned-to-unpruned forest
    acc_test_ratio : ratio between test accuracy of pruned-to-unpruned forest

    """
    X_train, X_test, y_train, y_test = data

    acc_test_forest = get_acc_score(y_test, forest.predict(X_test), predicttype)

    oob_indices = get_oob_indices(forest, X_train)

    out_ = get_alpha(forest, X_train, X_test, oob_indices, predicttype)
    alpha_list, OOB_leaves_id, test_leaves_id, node_score, min_alpha_list = out_

    """ get CV error for tree t individually by setting optimum alpha[t]"""
    opt_alpha = get_opt_alpha(forest, OOB_leaves_id, y_train, alpha_list, oob_indices, 
                              node_score, predicttype)
#    """ get train error for all trees by setting optimum alpha gloablly"""
#    opt_alpha_oob = get_glob_opt_alpha(forest, y_train, oob_indices, train_leaves_id, 
#                                       alpha_list, node_score, predicttype)
    get_glob_thresh_CV_plot(forest, X_train, y_train, X_test, y_test, 
                            node_score, predicttype)
    """ get CV error for all trees by setting optimum alpha gloablly"""
    opt_alpha_oob = get_glob_opt_alpha(forest, y_train, oob_indices, OOB_leaves_id, 
                                       alpha_list, node_score, predicttype)
    """ (tree) Prune up to optimal alphas """
    opt_test_id, prune_ratio = get_optpruned_tree(forest, test_leaves_id, 
                                                    opt_alpha, min_alpha_list)
    y_pruned = predict_forest(node_score, opt_test_id, predicttype)
    acc_test_pruned = get_acc_score(y_test, y_pruned, predicttype)
    """ (Forest) Prune up to optimal alphas """
    opt_test_id_oob, prune_ratio_oob = get_optpruned_tree(forest, test_leaves_id, 
                                                          opt_alpha_oob, 
                                                          min_alpha_list)
    y_pruned_oob = predict_forest(node_score, opt_test_id_oob, predicttype)
    acc_test_pruned_oob = get_acc_score(y_test, y_pruned_oob, predicttype)

    def plot_alpha_lists():
        plt.figure()
        for t in alpha_list:
            plt.plot(alpha_list[t])
        plt.xlabel('Number of unique' +r'$\alpha \in \mathcal{A}_j$')
        plt.ylabel( 'Cost complexity parameter' + r'$\mathcal{A}_j$')
        #comment save plots and reduce number of iterations to 1 and proceed
        #with model names etc
        #TODO conditionally return different results!

    acc_test_ratio = acc_test_pruned/acc_test_forest
    acc_test_ratio_oob = acc_test_pruned_oob/acc_test_forest
    """ V. Results """
    display_dict = {
            '-> Accuracy ratio : Tree Vs OOB : ' : repr(acc_test_ratio)+' VS ' + 
                                                   repr(acc_test_ratio_oob),
            '-> Size ratio     : Tree Vs OOB : ' :  repr(prune_ratio)+' VS ' + 
                                                    repr(prune_ratio_oob),
            '-> Pruned VS Original Test Accuracy    : ' : repr(acc_test_pruned) 
                                                    +' VS '+repr(acc_test_forest),
            '-> Pruned/Original Test Accuracy ratio : ' : repr(acc_test_ratio),
            '-> Pruned/Original Forest size ratio   : ' : repr(prune_ratio)
            }
    print('')
#    for dispkeys in display_dict:
#        print(dispkeys+display_dict[dispkeys])

    """Calculate CV on OOB prediction, prune and evalute accuracy """
    def CV_on_OOB_prediction():

        print('CV using OOB-prediction error')
        opt_index, opt_alpha = get_opt_alpha_OOB(forest, y_train, node_list, 
                                                node_score, oob_indices, 
                                                oob_leaves_id, alpha_list,
                                                predicttype)

        test_leaves_id, opt_nl_oob = get_optpruned_tree(forest, X_test, 
                                                        node_list, opt_index)
        prune_ratio_oob=np.mean(opt_nl_oob)
        y_pred_test_oob = predict_forest(node_score, test_leaves_id, predicttype)
        acc_test_pruned_oob = get_acc_score(y_test, y_pred_test_oob, predicttype)
        acc_test_ratio_oob = acc_test_pruned_oob/acc_test_forest
        print('')
        print('---------------OOB---------------')
        print('-> Pruned VS Original Test Accuracy    : ' + 
              repr(acc_test_pruned_oob)+' VS '+repr(acc_test_forest))
        print('-> Pruned/Original Test Accuracy ratio : ' + 
              repr(acc_test_ratio_oob))
        print('-> Pruned/Original Forest size ratio   : ' + 
            repr(prune_ratio_oob))
    # CV_on_OOB_prediction()
    output_array = np.array([prune_ratio, acc_test_ratio, prune_ratio_oob,
                             acc_test_pruned_oob])
    alpha_vals = np.concatenate((opt_alpha, opt_alpha_oob))
    return np.concatenate((output_array, alpha_vals))


def _get_data(dataset, test_size, predicttype):
    """ Function reads dataset with test_size proportion for 
    predicttype = 'classify/regress' tasks """

    X, y = get_dataset(dataset, predicttype)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)
    data = X_train, X_test, y_train, y_test
    return data

def _get_forest(model_name, n_trees, data, predicttype):
    """ Function creats decision tree ensemble for 
    predicttype = 'classify/regress' tasks """
    X_train, X_test, y_train, y_test = data
    forest = get_models(model_name, predicttype)
    forest.set_params(n_estimators = n_trees)
    forest.fit(X_train, y_train)
    return forest

def test_prune_classifiers(n_iter, n_cores, models, all_datasets, n_trees,
                           all_predictiontypes, CV_set='OOB-set'):
    """ Main test function for all tree-ensemble based classifiers
    - On OOB set as CV set the test accuracy and size ratio is better
    compared to train set as CV set

    Inputs
    ------
    predicttype : 'classify'/'regress'
    CV_set      : 'OOB-set'/'train-set'
    """

    orig_stdout = sys.stdout
    f = open('results/out.txt', 'w')
    sys.stdout = f
    if not os.path.exists(iters_result_path):
        os.makedirs(iters_result_path)
    for predicttype in all_predictiontypes: #classify/regress
        for dataset in all_datasets[predicttype]:
            test_size = get_split_sizes(dataset, predicttype)
            size_mat = np.zeros(shape=(len(models)*2,n_iter))
            acc_mat = np.zeros(shape=(len(models)*2,n_iter))
            alpha_mat = np.zeros(shape=(len(models)*2,n_iter, n_trees))
            j = 0
            all_models = []
            out_dict = {}
            for m, model in enumerate(models):
                arg_list = []
                for i in range(n_iter):
                    data = _get_data(dataset, test_size, predicttype)
                    forest = _get_forest(model, n_trees, data, predicttype)
                    argtuple = (CCpruning_withCV, forest, data, CV_set, 
                                predicttype)
                    arg_list.append(argtuple)
                """ parallel loop """
                results = Parallel(n_jobs=n_cores)(delayed(helper_parallel)(args)
                                                    for args in arg_list)

                results_mat = np.array(results)

                size_mat[j,:] = results_mat[:,0]
                acc_mat[j,:] = results_mat[:,1]
                val1_mean = "{0:.3f}".format(results_mat[:,0].mean())
                val1_std =  "{0:.3f}".format(results_mat[:,0].std())
                val1 =  val1_mean + '+/-' + val1_std
                val2_mean = "{0:.3f}".format(results_mat[:,1].mean()) 
                val2_std = "{0:.3f}".format(results_mat[:,1].std()) 
                val2 = val2_mean + '+/-' + val2_std
                out_dict[model+'-Tree'] = val1 +',   '+ val2
                alpha_mat[j,:,:] = results_mat[:,4:4+n_trees]
                j+=1
                all_models.append(model+'\n'+'Tree')
                size_mat[j,:] = results_mat[:,2]
                acc_mat[j,:] = results_mat[:,3]
                val1_mean = "{0:.3f}".format(results_mat[:,2].mean()) 
                val1_std = "{0:.3f}".format(results_mat[:,2].std())
                val1 = val1_mean + '+/-' + val1_std
                val2_mean = "{0:.3f}".format(results_mat[:,3].mean()) 
                val2_std = "{0:.3f}".format(results_mat[:,3].std()) 
                val2 = val2_mean + '+/-' + val2_std
                out_dict[model+'-Forest'] = val1 +',   '+ val2
                alpha_mat[j,:,:] = results_mat[:,4+n_trees::]
                j+=1
                all_models.append(model+'\n'+'OOB')

#            alpha_mat = alpha_mat.mean(axis=1)
            plot_vals = size_mat, acc_mat, alpha_mat

            f_name = iters_result_path + 'perf_plot_ALL_'+dataset
            if(n_iter>10):
                plot_scores(all_models, plot_vals, n_trees, dataset, f_name )
            print('')
            print(dataset)
            print('-----------------------------------------------')
            for keys in out_dict:
                print(keys + '==> Size, Acc ratio  : ' + out_dict[keys])

    sys.stdout = orig_stdout
    f.close()
    return

debug_flag = 0
result_path = './results/CC_prune/'
iters_result_path = './results/CC_prune/size_accuracy_rations/'

if __name__ == '__main__':
    """ parameters for test_prune_classifiers() """
    test_prune_classifiers_params = {
    'models' : ['RandomForest'], #, 'ExtraTrees', 'Bagger'], # 'RandomSubspace']
    'all_datasets' : {
            'classify' : ['iris'], #, 'digits', 'redwine', 'whitewine'],
            'regress' : ['boston']},
    'all_predictiontypes' : ['classify'], #'classify'
    'n_iter' : 12, 
    'n_cores' : 4,
    'n_trees' : 200
    }
    print('')
    test_prune_classifiers(**test_prune_classifiers_params)

    # test_prepruning()

    # test_plot_CCpruning_parameters()

    # simple_prune_demo()
    
    # test_regress_forest()

#    plot_ISMM_sub()

#    plot_tree_CV_error()
#    a_prune, a_mean = test_over_all_models_datasets()

