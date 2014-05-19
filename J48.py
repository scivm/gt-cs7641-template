import sys

import java.io.FileReader as FileReader
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean

import weka.core.Instances as Instances
import weka.classifiers.trees.J48 as J48
import weka.classifiers.Evaluation as Evaluation
import weka.core.Range as Range

"""
A simple example of using Weka classifiers (i.e., J48) from within Jython.

Commandline parameter(s):

    first parameter must be the ARFF file one wants to process with J48

"""

# check commandline parameters
if (not (len(sys.argv) == 2)):
    print "Usage: UsingJ48.py <ARFF-file>"
    sys.exit()

# load data file
print "Loading data..."
file = FileReader(sys.argv[1])
data = Instances(file)

# set the class Index - the index of the dependent variable
data.setClassIndex(data.numAttributes() - 1)

# create the model
print "Training J48..."
j48 = J48()
j48.buildClassifier(data)

# print out the built model
print "Generated model:\n"
print j48

evaluation = Evaluation(data)
buffer = StringBuffer()             # buffer for the predictions
attRange = Range()                  # no additional attributes output
outputDistribution = Boolean(False) # we don't want distribution
evaluation.evaluateModel(j48, data, [buffer, attRange, outputDistribution])
print evaluation.toSummaryString()


