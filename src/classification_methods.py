from sklearn import tree
from run_state_keeper import StateKeeper
import graphviz


def getDecisionTree(data_x=[0], data_y=[0], test_x=[0], test_y=[0], attributeNames=[''], split=1, depth=1) :
    run = str(StateKeeper.i)
    dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=split, max_depth=depth)
    dtc = dtc.fit(data_x,data_y)
    test = dtc.predict(test_x)
    print(test)
    print(test_y)
    out = tree.export_graphviz(dtc, out_file='tree_gini_run_'+run, feature_names=attributeNames)
    graphviz.render('dot','png','tree_gini_run_'+run,quiet=False)
    src=graphviz.Source.from_file('tree_gini_run_'+run)
    StateKeeper.i += 1
    return 0

def resetStateHolder(): 
    StateKeeper.i = 1