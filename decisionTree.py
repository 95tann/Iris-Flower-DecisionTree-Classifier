# *************************************************************************
#  Topic  : Learning Decision Trees
#  File   : decisionTree.py
#  Author : Tanmay
# *************************************************************************
from sklearn import tree
from sklearn.tree import export_graphviz

import graphviz
import pydotplus
import pandas as pd
import io

table_headers = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
df = pd.read_csv("set_a.csv", names = table_headers)
#print(df)

# Fetures & Labels
X = df[['sepal_len','sepal_width','petal_len', 'petal_width']]
Y = df[['class']]

# decision tree classifier
clf = tree.DecisionTreeClassifier(criterion='entropy')

#train classifer
clf = clf.fit(X,Y)

#build decision tree
features_list = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width']
def build_tree(tree, features, output_path):
	f = io.StringIO()
	export_graphviz(tree, out_file=f, feature_names=features)
	pydotplus.graph_from_dot_data(f.getvalue()).write_png(output_path)

# run program
def main():
	try:
		image_file = input("Enter filename to store decision tree output: ")
		image_file = image_file + '.png'
		build_tree(clf,features_list,image_file)
		print("Success! Output stored in file %s" % image_file)
	except:
		print("Something went wrong. Try again")
		exit()
main()
