#!/bin/sh
# Simple example command line calls to weka
# for 5 different supervised learning methods
export CLASSPATH=./lib/weka.jar:$CLASSPATH

# J48 Decision Tree
java -classpath .:lib/weka.jar weka.classifiers.trees.J48 -t data/iris.arff

# Support Vector Machine
java -classpath .:lib/weka.jar weka.classifiers.functions.LibSVM -t data/iris.arff

# K- Nearest Neighbor
java -classpath .:lib/weka.jar weka.core.neighboursearch.LinearNNSearch -t data/iris.arff

# ADA Boost
java -classpath .:lib/weka.jar weka.classifiers.meta.AdaBoostM1 -t data/iris.arff

# Neural Network
java -classpath .:lib/weka.jar weka.classifiers.functions.MultilayerPerceptron -t data/iris.arff
