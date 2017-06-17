# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 15:11:32 2017

@author: ravikiran
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score
from sklearn.ensemble.forest import _generate_sample_indices,\
                                    _generate_unsampled_indices
from utils import drawProgressBar
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy
import os

le = preprocessing.LabelEncoder()

def get_leaves_at_node(i, c_l, c_r):
    """ return id of leaves below this node i """
    stack = [i]
    n_l = []
    i_n = []
    while len(stack) > 0:
        i = stack.pop()
        # If we have a test node
        if (c_l[i] != c_r[i]):
            stack.append(c_l[i])
            stack.append(c_r[i])
            i_n.append(c_l[i])
            i_n.append(c_r[i])
        else:
            n_l.append(i)
        i_n = list(set(i_n)-set(n_l))
    return n_l, i_n

def prune_list(i, c_l, c_r):
    """ removes (children set to -1) internal node at 
    node id i from both r and l children list """
    cc_l = list(c_l)
    cc_r = list(c_r)
    stack = [i]
    while len(stack)>0:
        i = stack.pop()
        if(cc_l[i]!=cc_r[i]):
            stack.append(cc_l[i])
            stack.append(cc_r[i])
            cc_l[i] = -1
            cc_r[i] = -1
    return cc_l, cc_r

def predict_tree(node_score, tree_leaves_id, predicttype):
    """ predict tree for classify and regress tasks """
    if(predicttype=='classify'):
        prob_val = node_score[tree_leaves_id, :]
        y_predicted = np.argmax(prob_val, axis=1)
    else:
        y_predicted = node_score[tree_leaves_id]
    return y_predicted

def predict_forest_oob(node_score, oob_indices, oob_leaves_id, n_samples, predicttype):
    """
    Function calculates the OOB prediction given the indicator matrix and
    leaves id for each tree
    """
    n_trees = len(node_score)
    if(predicttype=='classify'):
        num_classes = node_score[0].shape[1]
        prob_oob = np.zeros(shape=(n_samples, num_classes))
        for t in range(n_trees):
            prob_oob[oob_indices[t], :] += node_score[t][oob_leaves_id[t],:]
        y_oob = np.argmax(prob_oob, axis=1)
    else:
        y_oob = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)
        for t in range(n_trees):
            y_oob[oob_indices[t]] += node_score[t][oob_leaves_id[t]]
            n_predictions[oob_indices[t]] += 1
        n_predictions[n_predictions == 0] = 1
        y_oob /= n_predictions
    return y_oob

def predict_forest(node_score, test_leaves_id, predicttype):
    """
    Function implements predict for forest regressor/classifier

    Inputs
    ------
    node_score : class probabilities / mean values
    predicttype : classify/regress
    test_size : test sizes

    Outputs
    -------
    y_pred_forest : predicted output of forest regressor/classifier
    """


    n_trees = len(node_score)
    test_size = test_leaves_id[0].shape[0]

    if(predicttype=='classify'):
        num_classes = node_score[0].shape[1]
        prob_val_test = np.zeros(shape=(test_size, num_classes))
        for t in range(n_trees):
            prob_val_test += node_score[t][test_leaves_id[t], :]

        prob_val_test = prob_val_test/n_trees
        y_pred_forest = np.argmax(prob_val_test, axis=1)
    else:
        mean_val_test = np.zeros(shape=(test_size))
        for t in range(n_trees):
            mean_val_test += node_score[t][test_leaves_id[t]]

        y_pred_forest = mean_val_test/n_trees

    return y_pred_forest

def _update_leaves(node, c_l, c_r, leaves_id):
    """Updates leaves to pruned leaves. Pruned leaves are obtained
    by removing the leaves under node 'node'
    Inputs
    ------
    node: node id whose leaves are pruned and updated c_l and c_r
    c_l : children left
    c_r : children right
    leaves_id : samples labelled with their corresponding leaf nodes

    Outputs
    -------
    c_l : new children left after pruning
    c_r : new children right after pruning
    
    """
    c_l_, c_r_ = prune_list(node, c_l, c_r)
    leaves, _ = get_leaves_at_node(0,c_l_,c_r_)
    for l in leaves:
        old_l, _ = get_leaves_at_node(l,c_l,c_r)
        for leaf in old_l:
            leaves_id[leaves_id==leaf] = l
    c_l, c_r = c_l_, c_r_
    return c_l, c_r, leaves_id, leaves

def get_node_means(node_indicator_t, y_train_t):
    """ Get mean values at all nodes in the tree : for regression task """
    n_nodes = node_indicator_t.shape[1]
    mean_vals = np.zeros(n_nodes)
    exp_var = np.zeros(n_nodes)
    for node in range(n_nodes):
        node_idx, _ = node_indicator_t[:,node].nonzero()
        mean_vals[node] = np.mean(y_train_t[node_idx])
        exp_var[node] = explained_variance_score(y_train_t,
                     mean_vals[node]*np.ones(len(y_train_t)))

    return mean_vals, exp_var

def get_class_prob(estimator):
    """ Gets the class probabilities in estimator for all nodes once
    For Classify task
    
    old implementation : 
    n_nodes = estimator.tree_.node_count
    node_indicator = estimator.decision_path(X_train)
    num_classes = estimator.n_classes_
    class_prob = np.zeros(shape=(n_nodes, num_classes))
    for i in range(n_nodes):
        s,_ = node_indicator[:,i].nonzero()
        p = np.bincount(y_train[s])/len(s)
        class_prob[i, :len(p)] = p
        return class_prob
    """
    bin_count = estimator.tree_.value[:,0,:]
    bin_count /= bin_count.sum(axis=1)[:,np.newaxis]
    return bin_count
    

def get_acc_score(y_pred, y_true, predicttype):
    """ calculates accuracy for classification and 
    r2 score/exp variance for regression"""
    if(predicttype=='classify'):
        acc_ = accuracy_score(y_true, y_pred)
    else:
        #has to be an accuracy : 0 worst and 1 best
        acc_ = explained_variance_score(y_true, y_pred)
        #acc_ = r2_score(y_true, y_pred)
    return acc_

def calculate_alpha(r, p, internal_nodes, left, right):
    """ calculate the cost-complexity parameter for `nodes in tree """
    alpha = {}
    for node in internal_nodes:
        l_node, _ = get_leaves_at_node(node, left, right)
        R_T = r[node] * p[node]
        R_T_t = np.dot(r[l_node], p[l_node])
        alpha[node] = (R_T - R_T_t)/ (len(l_node)-1)

    return alpha

def get_error_function(estimator):
    """
    Calculates the resubstitution error for each node in the decision tree
    Input
    -----
    estimator : decision tree
    
    Output
    ------
    Resubstitution error for each node

    old implementation : 
    
    n_nodes = estimator.tree_.node_count
    r = np.array([0]*n_nodes,dtype=float)
    M = estimator.tree_.value[:,0,:].view()
    for node in range(n_nodes):
        vals = M[node,:]
        total = sum(vals)
        vals[np.argmax(vals)] = 0
        error_sum = sum(vals)
        r[node] = error_sum/total    
    """
    return estimator.tree_.impurity

def get_oob_indices(forest, X_train):
    oob_indices = {}
    for t, estimator in enumerate(forest):
        oob_indices[t] = _generate_unsampled_indices(estimator.random_state,
                                                      X_train.shape[0])
    return oob_indices

def _prune_tree(tree_params, oob_id, test_id):
    """ Given node scores and leaves id, prunes current tree in c_l, c_r
    Inputs : tree structure, tree_leaves_id
    Outputs : tree_leaves_id
    """
    c_l, c_r, r, p  = tree_params
    leaves, i_nodes = get_leaves_at_node(0, c_l, c_r)
    alpha_list, node_list = [], []
    all_oob_id, all_test_id = {}, {}
    temp_root_oob = dcopy(oob_id)
    temp_root_test = dcopy(test_id)
    #leaves id matrix : #samples \times #alphas
    while(len(i_nodes)):
            alpha = calculate_alpha(r, p, i_nodes, c_l, c_r)
            if(alpha):
                min_alpha_node = min(alpha, key=alpha.get)
                min_alpha = alpha[min_alpha_node]
                alpha_list.append(min_alpha)
                node_list.append(min_alpha_node)
                c_l_, c_r_ = prune_list(min_alpha_node, c_l, c_r)
                leaves_, i_nodes = get_leaves_at_node(0,c_l_,c_r_)
                #This can be used to label test samples
                for l in leaves_:
                    old_l, _ = get_leaves_at_node(l,c_l,c_r)
                    for leaf in old_l:
                        oob_id[oob_id==leaf] = l
                        test_id[test_id==leaf] = l
                all_oob_id[min_alpha] = dcopy(oob_id)
                all_test_id[min_alpha] = dcopy(test_id)
                c_l, c_r = c_l_, c_r_
            else:
                print('Done with pruning')
                break
            
    epsilon = 1e-10 #internal parameter
    least_alpha = min(alpha_list) - epsilon
    all_oob_id[least_alpha] = temp_root_oob
    all_test_id[least_alpha] = temp_root_test
    alpha_list = [least_alpha] + alpha_list
    return alpha_list, all_oob_id, all_test_id

def get_alpha(forest, X_train, X_test, oob_indices, predicttype):
    """ Calculate the sequence of alphas corresponding to nested trees
    for each estimator in forest.
    alpha_list : set of alpha values
    acc_list_train : training set accuracy after each pruning step (alpha)
    node_list : the set of nodes to prune for each alpha
    """
    node_score, OOB_leaves_id, alpha_list = {}, {}, {}
    n_trees = forest.n_estimators
    min_alpha_list = {}
    test_leaves_id = {}
    for t in range(n_trees):
#        drawProgressBar(t/n_trees)
        estimator = forest.estimators_[t]
        c_l = np.array(estimator.tree_.children_left, dtype=int)
        c_r = np.array(estimator.tree_.children_right, dtype=int)
        X_OOB_tree = X_train[oob_indices[t], :].view()
#        y_OOB_tree = y_train[oob_indices[t]].view()
        #substitution error and node occupation probablities
#        if(predicttype=='classify'):
        r = get_error_function(estimator)
#            indicator, n_nodes_ptr = forest.decision_path(X_train)
#            node_indicator = indicator[:,n_nodes_ptr[t]:n_nodes_ptr[t+1]]
#            mean_vals, r = get_node_means(node_indicator, y_OOB_tree)
#            r  = estimator.tree_.impurity
#            mean_vals = np.zeros(node_indicator.shape[1])
#            r = np.zeros(node_indicator.shape[1])
#            for node in range(node_indicator.shape[1]):
#                node_idx, _ = node_indicator[:,node].nonzero()
#                mean_vals[node] = np.mean(y_train[indicies][node_idx])
#                r[node] = explained_variance_score(y_train[indicies],
#                             mean_vals[node]*np.ones(len(indicies))) #get_node_means(node_indicator, y_train_tree)
        p = estimator.tree_.n_node_samples/estimator.tree_.n_node_samples[0]
       #scores
#        if(predicttype=='classify'):
        node_score[t] = get_class_prob(estimator)
#        else:
#            node_score[t] = mean_vals #get_node_means(estimator, X_train_tree, y_train_tree)

        oob_id = estimator.apply(X_OOB_tree)
        test_id = estimator.apply(X_test)
        tree_params = c_l, c_r, r, p
        alpha_list[t], OOB_leaves_id[t], test_leaves_id[t] = _prune_tree(tree_params, oob_id, test_id)
        min_alpha_list[t] = min(alpha_list[t])
    print('')
    return alpha_list, OOB_leaves_id, test_leaves_id, node_score, min_alpha_list

def get_opt_alpha(forest, OOB_leaves_id, y_train, alpha_list, oob_indices,
                  node_score, predicttype):
    """ calculate the optimum value of alpha given a sequence of nested trees
    and Cross validation set (either OOB or training)

    j is the index for trees in the forest
    1. \alpha*_j = \argmin_\alpha_j Error_j(OOB_set, j)
        (much smaller size and same accuracy)
    2. \alpha*_j = \argmin_\alpha_j Error_j(train_set, j)
        (same accuracy and size)
    3. \alpha*_j = \argmin_\alpha_j \sum_j OOB-Error(OOB_set, j) (same as 1)
    4. \alpha*_j = \argmin_\alpha_j \sum_j OOB-Error(train_set, j)
        (same accuracy and size)
    5. Can there be something in between 1,3 and 2,4
    """

    opt_alpha = []
#    n_trees = forest.n_estimators
    for t, estimator in enumerate(forest):
#        drawProgressBar(t/n_trees)
        acc_list_test = []
        for alpha in alpha_list[t]:
            leaves_id = OOB_leaves_id[t][alpha]
            y_predicted = predict_tree(node_score[t], leaves_id, predicttype)
            acc_ = get_acc_score(y_train[oob_indices[t]], y_predicted, predicttype)
            acc_list_test.append(acc_)
#        plt.figure()
#        plt.title('Tree#'+repr(t))
#        plt.plot(acc_list_test)
#        plt.title('Alpha index : ' +repr(np.argmax(acc_list_test)) 
#                 + '#values=' + repr(len(acc_list_test)))
#        plt.show()
        opt_index = np.argmax(acc_list_test)
        opt_alpha.append(alpha_list[t][opt_index])
    print('')
    return opt_alpha

def get_optpruned_tree(forest, alpha_OOB_leaves_id, opt_alpha, min_alpha):
    """
    Prune intial tree to optimal subtree using alpha obtained after
    cross-validation on (OOB-set of train-set)
    """
    n_trees = forest.n_estimators
    opt_nl = [0]*n_trees
    test_leaves_id = {}
    for t, estimator in enumerate(forest):
        test_leaves_id[t] = alpha_OOB_leaves_id[t][opt_alpha[t]]
        unpruned_leaves_id = alpha_OOB_leaves_id[t][min_alpha[t]]
        orig_nl = len(np.unique(unpruned_leaves_id))
        opt_nl[t] = len(np.unique(test_leaves_id[t]))/orig_nl

    return test_leaves_id, np.mean(opt_nl)

def get_glob_thresh_CV_plot(forest, X_train, y_train, X_test, y_test, 
                            node_score, predicttype):
    """ Returns training error plot for global thresholding """
    

    alpha_list, train_leaves_id, test_leaves_id = {}, {}, {}
    for t, estimator in enumerate(forest):
        estimator = forest.estimators_[t]
        c_l = np.array(estimator.tree_.children_left, dtype=int)
        c_r = np.array(estimator.tree_.children_right, dtype=int)
        r = get_error_function(estimator)
        p = estimator.tree_.n_node_samples/estimator.tree_.n_node_samples[0]
        node_score[t] = get_class_prob(estimator)
        train_id = estimator.apply(X_train)
        test_id = estimator.apply(X_test)
        tree_params = c_l, c_r, r, p
        out_ = _prune_tree(tree_params, train_id, test_id)
        alpha_list[t], train_leaves_id[t], test_leaves_id[t] = out_
        
    
    unique_alpha = []
    for t in alpha_list:
        unique_alpha = unique_alpha + alpha_list[t]
        alpha_list[t] = np.array(alpha_list[t])
    unique_alpha = np.array(unique_alpha)

    #internal parameter to control the discretization of alphas
    num_decimals = 10
    unique_alpha = np.round(unique_alpha, num_decimals)

    unique_alpha = np.unique(unique_alpha)        

    def get_leaves_id_at_thresh(a_thresh):
        """ 
        Given a set of oob_leaves_id indexed by alpha : 
        and given a global threshold on alpha : a_thresh
        return the leaves id that correspond to value of 
        """
        train_id = {}
        test_id = {}
        for t, estimator in enumerate(forest):
            alpha_val = alpha_list[t][alpha_list[t] <= a_thresh]
            if(sum(alpha_val)): #when alpha_val is empty return largest alpha
                alpha_t = max(alpha_val)
            else:#return the lowest alpha value that is the full tree
                alpha_t = min(alpha_list[t])
            
            train_id[t] = train_leaves_id[t][alpha_t]
            test_id[t] = test_leaves_id[t][alpha_t]
        return train_id, test_id
        
    acc_list_train = []
    acc_list_test = []
    for i, a in enumerate(unique_alpha):
        train_id, test_id = get_leaves_id_at_thresh(a)
        y_pred_train = predict_forest(node_score, train_id, predicttype)
        y_pred_test = predict_forest(node_score, test_id, predicttype)
        acc_train = get_acc_score(y_train, y_pred_train, predicttype)
        acc_test = get_acc_score(y_test, y_pred_test, predicttype)
        acc_list_train.append(acc_train)
        acc_list_test.append(acc_test)
    
    plotpath = 'results/CV_plots/' 

    if not os.path.exists(plotpath):
            os.makedirs(plotpath) 
    # if not os.path.exists(plotpath+'global_cv_01.png'):
    #         os.makedirs(plotpath)
    # fname, _ = os.path.splitext(ts_key)

    plt.figure()
    plt.plot(unique_alpha, acc_list_train, label='train')
    plt.plot(unique_alpha, acc_list_test, label='test')
    plt.title('Global CC-paramter vs Classification accuracy')
    plt.xlabel(r'$\alpha = \cup_j \mathcal{A}_j$') #Sorted global CC parameter
    plt.ylabel('Classification Accuracy')
    plt.legend(loc='best')
    plt.savefig(plotpath+'test.png')
    return 
    
    
    
def get_glob_opt_alpha(forest, y_train, oob_indices, alpha_OOB_leaves_id,
                        alpha_list, node_score, predicttype):
    """ Function calculates one CC-parameter \alpha for all trees,
    while evaluating the OOB prediction error on the forest

    We plots training Error Vs validation error for random forest pruning
    - This is only possible for global pruning of random forests
    - The argmin of this error is already given by function
    """


    unique_alpha = []
    for t in alpha_list:
        unique_alpha = unique_alpha + alpha_list[t]
        alpha_list[t] = np.array(alpha_list[t])
    unique_alpha = np.array(unique_alpha)

    #internal parameter to control the discretization of alphas
    num_decimals = 10
    unique_alpha = np.round(unique_alpha, num_decimals)

    unique_alpha = np.unique(unique_alpha)

    train_size = len(y_train)

    acc_list_test = []
    level_index = {}
    
    for t, estimator in enumerate(forest):
        level_index[(0,t)] = 0
    
    def get_oob_leaves_id_at_alpha(a_thresh):
        """ 
        Given a set of oob_leaves_id indexed by alpha : 
        and given a global threshold on alpha : a_thresh
        return the leaves id that correspond to value of 
        """
        oob_leaves_id = {}
        alpha_keys = []
        for t, estimator in enumerate(forest):
            alpha_val = alpha_list[t][alpha_list[t] <= a_thresh]
            if(sum(alpha_val)): #when alpha_val is empty return largest alpha
                alpha_t = max(alpha_val)
            else:#return the lowest alpha value that is the full tree
                alpha_t = min(alpha_list[t])
            oob_leaves_id[t] = alpha_OOB_leaves_id[t][alpha_t]
            alpha_keys.append(alpha_t)
        return oob_leaves_id, alpha_keys
    
    all_alphas = {}
    for i, a in enumerate(unique_alpha):
#        drawProgressBar(i/len(unique_alpha))
        oob_leaves_id, all_alphas[i] = get_oob_leaves_id_at_alpha(a)
        y_pred_oob = predict_forest_oob(node_score, oob_indices, oob_leaves_id, 
                                        train_size, predicttype)
        acc_ = get_acc_score(y_train, y_pred_oob, predicttype)
        acc_list_test.append(acc_)

    opt_alpha = all_alphas[np.argmax(acc_list_test)]
#    plt.plot(acc_list_test, label='Test')
    
    return opt_alpha

def get_opt_alpha_OOB(forest, y_train, node_list, node_score, oob_indices,
                  oob_leaves_id, alpha_list, predicttype):
    """
    Function calculates optimal alpha/index over the whole forest using
    the oob prediction error for crossvalidation.
    """
    n_trees, train_size = forest.n_estimators, len(y_train)
    OOB_tree_indicator, opt_alpha, opt_index = {}, [0]*n_trees, [0]*n_trees
    #get OOB indicator for each sample in training set
    for s in range(train_size):
        OOB_tree_indicator[s] = []
        for t in range(n_trees):
            if(s in oob_indices[t]):
                OOB_tree_indicator[s].append(t)

    for t, estimator in enumerate(forest):
        drawProgressBar(t/n_trees)
        acc_list_test = []
        c_l = estimator.tree_.children_left
        c_r = estimator.tree_.children_right
        leaves, _ = get_leaves_at_node(0, c_l, c_r)
        for node in node_list[t]:
            y_pred_oob = predict_forest_oob(node_score, oob_indices, oob_leaves_id, 
                                            train_size, predicttype)
            acc_ = get_acc_score(y_train, y_pred_oob, predicttype)
            acc_list_test.append(acc_)
            c_l, c_r, oob_leaves_id[t], _ = _update_leaves(node, c_l, c_r,
                                                       oob_leaves_id[t])
        opt_index[t] = np.argmax(acc_list_test)
        opt_alpha[t] = alpha_list[t][opt_index[t]]
    return opt_index, opt_alpha


def idea_to_implement():
    print('-Calculate the test leaves id at the same time as train leaves id')
    print('Then predict with optimal leaf labeling')