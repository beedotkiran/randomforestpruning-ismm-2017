from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
# use seaborn plotting defaults
import seaborn as sns; sns.set()
from sklearn.ensemble.forest import _generate_sample_indices,\
                                    _generate_unsampled_indices
from prune_funcs import get_alpha, predict_forest_oob, get_acc_score
from sklearn.datasets import make_blobs, load_boston, load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from prune_funcs import get_node_means, get_class_prob, get_error_function
from utils import get_models
from utils import get_forest_size

import scipy.sparse as ssp

if __name__=='__main__':
#def test_matricielle_forest_prediction():
    n_samples = 100
    num_classes = 3
    X, y = make_blobs(n_samples=n_samples, centers=num_classes, random_state=2,
                      cluster_std=2.0)
#    y += 1
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
    estimator = DecisionTreeClassifier()
    estimator.fit(X_train, y_train)
    n_nodes = estimator.tree_.node_count
    bin_count = estimator.tree_.value[:,0,:]
    I_j = estimator.decision_path(X_test)
    p = estimator.tree_.n_node_samples/X_train.shape[0]
    class_prob = get_class_prob(estimator)
    bin_count /= bin_count.sum(axis=1)[:,np.newaxis]
    class_prob_diff = np.linalg.norm(class_prob- bin_count)
    print('clas prob diff = '+repr(class_prob_diff))
#    y_test += 1
    y_test_eye = ssp.lil_matrix((len(y_test), len(y_test)), dtype=np.int)
    y_test_eye.setdiag(y_test)
    W_j = I_j.T*y_test_eye
    leaves_id = estimator.apply(X_test)
#    leaves_id += 1
#    L_j = W_j[leaves_id, :]
#    B_j = np.zeros(shape=(len(y_test), num_classes+1), dtype=np.int)
#    for i in range(L_j.shape[0]):
#        _, _, ys = ssp.find(L_j[i,:])
#        B_j[i,:] = np.bincount(ys, minlength=num_classes+1)
    y_pred = bin_count[leaves_id].argmax(axis=1) #B_j.argmax(axis=1)
    y_est = estimator.predict(X_test)
    diff = np.linalg.norm(y_pred-y_est)
    print('diff='+repr(diff))
    #http://stackoverflow.com/questions/19201972/can-numpy-bincount-work-with-2d-arrays
#    B_j = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_classes), axis=1, arr=W_j)

    
    
def test_resuberror_classifier():
#    iris = load_iris()
#    X = iris.data  # we only take the first two features.
#    y = iris.target
    X, y = make_blobs(n_samples=n_samples, centers=num_classes, random_state=2,
                      cluster_std=2.0)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,
                                                        random_state=0)
    estimator = DecisionTreeClassifier()
    estimator.fit(X_train, y_train)
    imp_est = estimator.tree_.impurity
    r = get_error_function(estimator)
    plt.figure()
    plt.plot(r, label='mine')
    plt.plot(imp_est, label='impurity')
    plt.legend(loc='best')
    plt.show()
    return r, imp_est

def test_tree_regress():
    """ test to predict for single tree """
    boston = load_boston()
    X = boston.data
    y = boston.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,
                                                        random_state=0)

    estimator = DecisionTreeRegressor()
    estimator.fit(X_train, y_train)
    node_indicator = estimator.decision_path(X_train)
    mean_vals, _ = get_node_means(node_indicator, y_train)
    y_pred_dt = estimator.predict(X_test)
    test_leaves_id = estimator.apply(X_test)
    y_pred_mine_dt = mean_vals[test_leaves_id]
    diff = np.linalg.norm(y_pred_dt-y_pred_mine_dt)
    print('Tree predictions diff :'+repr(diff))
    return

def test_class_prob():
    """ testing class probabilities from Random forests """
    n_trees = 100
    num_classes = 20
    X, y = make_blobs(n_samples=1000, centers=num_classes, random_state=2,
                      cluster_std=2.0)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,
                                                        random_state=0)
    forest = get_models('RandomForest', 'classify')
    forest.set_params(n_estimators=n_trees)
    forest.fit(X_train, y_train)
    y_pred_forest = forest.predict(X_test)
    prob_val_all = np.zeros(shape=(len(y_test), num_classes))
    n_samples = X_train.shape[0]
    print('Diff over all trees :')
    for t, estimator in enumerate(forest):
        sample_indices = _generate_sample_indices(estimator.random_state, n_samples)
        y_tree_predict = estimator.predict(X_test)
        class_prob = get_class_prob(estimator)
        test_leaves_id = estimator.apply(X_test)
        y_tree_mine =  class_prob[test_leaves_id,:]
        diff = np.linalg.norm(y_tree_predict-np.argmax(y_tree_mine, axis=1))
        print("%.2f" % round(diff,2), end=', ')
        prob_val_all +=  y_tree_mine#n_nodes, num_classes
    print('')
    prob_val_all = prob_val_all/n_trees
    y_pred_mine_rf = np.argmax(prob_val_all, axis=1)
    print('% Predictions diff = ')
    print(np.linalg.norm(y_pred_forest-y_pred_mine_rf))
    return

def test_regress_forest():
    """ testing Random forests regression predict function """
    n_trees = 4
    boston = load_boston()
    X = boston.data
    y = boston.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,
                                                        random_state=0)

#    X_train = np.array([range(1,4),range(4,7)])
#    y_train = np.array([9,5])
#    X_test = X_train
#    y_test = y_train
    print('Single regression tree test : ')
    estimator = DecisionTreeRegressor()
    estimator.fit(X_train, y_train)
    y_pred_dt = estimator.predict(X_test)

    node_indicator = estimator.decision_path(X_train)
    mean_vals, _ = get_node_means(node_indicator, y_train)

    test_leaves_id = estimator.apply(X_test)
    y_pred_mine_dt = mean_vals[test_leaves_id]
    diff = np.linalg.norm(y_pred_dt-y_pred_mine_dt)
    print('Tree predictions diff :'+repr(diff))

    print('Regression Forest Test : ')
    forest = get_models('RandomForest', 'regress')
    forest.set_params(n_estimators=n_trees)
    forest.fit(X_train, y_train)
    y_pred_all = np.zeros(shape=(len(y_test)))
    n_samples = X_train.shape[0]
    indicator, n_nodes_ptr = forest.decision_path(X_train)
    for t, estimator in enumerate(forest):
        t_idx = _generate_sample_indices(estimator.random_state, n_samples)
        y_tree_predict = estimator.predict(X_test)
        print('Num nodes = '+repr(estimator.tree_.node_count))
        node_indicator = indicator[:,n_nodes_ptr[t]:n_nodes_ptr[t+1]]
#        node_indicator = estimator.decision_path(X_train)
        mean_vals, _ = get_node_means(node_indicator, y_train[t_idx])
        leaves_id = estimator.apply(X_test)
        y_tree_mine = mean_vals[leaves_id]
        diff = np.linalg.norm(y_tree_predict-y_tree_mine)
#        print(y_tree_predict, y_tree_mine)
        print('Tree#'+repr(t)+': Diff = '+repr(diff))
        y_pred_all += y_tree_mine
    y_pred_rf = forest.predict(X_test)
    y_pred_mine_rf = y_pred_all/n_trees
    diff = np.linalg.norm(y_pred_rf-y_pred_mine_rf)
    print('Forest predictions difference :' + repr(diff))
    print('#BUG#-->Trees in the forest dont match my tree predictions')
    return

def test_oob_prediction():
    """ tests if manual prediction is equal to the forest oob prediction """
    return

if __name__=='__main2__':
    n_trees=100
    num_classes = 3
    n_samples = 80
    predicttype='regress'
    #data for both classification and regression
    X_train = np.random.rand(n_samples,10)
    y_train = np.random.randint(num_classes,size=(n_samples))

    if(predicttype=='classify'):
        forest = RandomForestClassifier(n_estimators=n_trees, oob_score=True)
    else:
        forest = RandomForestRegressor(n_estimators=n_trees, oob_score=True)

    oob_indices, oob_leaves_id, OOB_tree_indicator = {}, {}, {}
    #fit
    forest.fit(X_train, y_train)
    forest_oob_score = forest.oob_score_

    n_trees, train_size = forest.n_estimators, len(y_train)
    indicator, n_nodes_ptr = forest.decision_path(X_train)
    node_indicator = {}
    sample_index = {}
    for t, estimator in enumerate(forest):
        oob_indices[t] = _generate_unsampled_indices(estimator.random_state,
                                                      X_train.shape[0])
        oob_leaves_id[t] = estimator.apply(X_train[oob_indices[t], :])
        sample_index[t] = _generate_sample_indices(estimator.random_state,n_samples)
        node_indicator[t] = indicator[:,n_nodes_ptr[t]:n_nodes_ptr[t+1]]
    mean_vals = {}
    for t in range(n_trees):
        mean_vals[t] = np.zeros(node_indicator[t].shape[1])
        for node in range(node_indicator[t].shape[1]):
            r, c = node_indicator[t][:,node].nonzero()
            mean_vals[t][node] = np.mean(y_train[sample_index[t]][r])

    alpha_list, _, node_score = get_alpha(forest, X_train, y_train, predicttype)

    y_pred_oob = np.zeros(len(y_train))
    print('Forest size, trees : '+repr(get_forest_size(forest))+','+repr(n_trees))
    if(predicttype=='classify'):
        print('---- Classify Task ----')
        predictions = np.zeros(shape=(len(y_train), num_classes))
        for t, estimator in enumerate(forest):
            pred_rf_est = estimator.predict_proba(X_train[oob_indices[t], :])
            #this does not give class probs as expected
            #see here https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/tree/tree.py#L770
            pred_mine_est = node_score[t][oob_leaves_id[t],:]
            diff_tree = np.linalg.norm(pred_rf_est-pred_mine_est)
            print("%.2f" % round(diff_tree,2), end=', ')
            predictions[oob_indices[t], :] += pred_mine_est
        print('')
        forest_oob_mine = np.mean(y_train == np.argmax(predictions, axis=1))
    else:
        print('---- Regression Task ----')
        predictions = np.zeros(len(y_train))
        n_predictions = np.zeros(len(y_train))
        print('Tree diffs : ')
        for t, estimator in enumerate(forest):
            pred_rf_est = estimator.predict(X_train[oob_indices[t], :])
            pred_mine_est = node_score[t][oob_leaves_id[t]]
            diff_tree = np.linalg.norm(pred_rf_est-pred_mine_est)
            print("%.2f" % round(diff_tree,2), end=', ')
            predictions[oob_indices[t]] += pred_mine_est
            n_predictions[oob_indices[t]] += 1
        print('')
        n_predictions[n_predictions == 0] = 1
        predictions /= n_predictions
        diff_oob = np.linalg.norm(forest.oob_prediction_-predictions)
        print('Diff. b/w forest-vs-mine OOB predictions :'+repr(diff_oob))
        forest_oob_mine = r2_score(y_train,predictions)

    print('OOB scores')
    print('Mine : ' +repr(forest_oob_mine))
    print('RFimp: ' +repr(forest_oob_score))


