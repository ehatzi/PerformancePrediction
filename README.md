PerformancePrediction
=====================

Implements a regression model for predicting the performance gain of a 
bootstrapping process prior to actually applying it. Computes the two 
features; baseline classifier maturity and oracle reliability.

Based on paper "PERFORMANCE PREDICTION OF BOOTSTRAPPING FOR IMAGE
CLASSIFICATION", Chatzilari et al.

Run PerformancePrediction.m to reproduce the results of the paper:


Input - Calculate features from visual and textual kernel

VisualAnnotationFileName & TextualAnnotationFileName: Files containing
the training set annotations; anns: matrix of dimensions #training
instances x # concepts with '1's wherever an image is positive for a
concept and -1 otherwise.

VisualKernelFileName file including the training kernel for calculating
the maturity feature - based on visual features

TextualKernelFileName: file including the training kernel for calculating
the reliability feature - based on the textual features, i.e. oracle



Input - Give features as input

maturityFileName: file including the maturity of a classifier as a matrix
of dimensions #concepts x # CV folds (matrix name should be AP)

maturityFileName: file including the reliability of a classifier as a
matrix of dimensions #concepts x # CV folds (matrix name should be AP)

Results_Initial.mat & Results_Final.mat: files containing the average
precision of the baseline and enhanced classifiers respectively. They
include a vector AP of dimension #concepts x 1



Download the data for a demo run from:
http://mklab.iti.gr/project/PerformancePrediction 

PRE-REQUIREMENTS:
Runs with matlab version of the libsvm library: the svmtrain and
svmpredict mex files should be added in the matlab path

Tested on Matlab 2012a

For any comments, questions, suggestions contact:
Elisavet Chatzilari ehatzi@iti.gr
