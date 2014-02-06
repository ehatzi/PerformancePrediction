function PerformancePrediction
%% function description
% Computes classifier maturity and oracle reliability, trains a regression
% model, calculates leave-one-out estimates and plots the results.
% Based on paper "PERFORMANCE PREDICTION OF BOOTSTRAPPING FOR IMAGE
% CLASSIFICATION", Chatzilari et al.
% 
% Input - Calculate features from visual and textual kernel
% VisualAnnotationFileName & TextualAnnotationFileName: Files containing
% the training set annotations; anns: matrix of dimensions #training
% instances x # concepts with '1's wherever an image is positive for a
% concept and -1 otherwise.
% VisualKernelFileName file including the training kernel for calculating
% the maturity feature - based on visual features
% TextualKernelFileName: file including the training kernel for calculating
% the reliability feature - based on the textual features, i.e. oracle
% 
% Input - Give features as input
% maturityFileName: file including the maturity of a classifier as a matrix
% of dimensions #concepts x # CV folds (matrix name should be AP)
% maturityFileName: file including the reliability of a classifier as a
% matrix of dimensions #concepts x # CV folds (matrix name should be AP)
% 
% Results_Initial.mat & Results_Final.mat: files containing the average
% precision of the baseline and enhanced classifiers respectively. They
% include a vector AP of dimension #concepts x 1
% 
% 
% Download the data for a demo run from:
% http://mklab.iti.gr/project/PerformancePrediction 
% 
% PRE-REQUIREMENTS:
% Runs with matlab version of the libsvm library: the svmtrain and
% svmpredict mex files should be added in the matlab path
% 
% Tested on Matlab 2012a
% For any comments, questions, suggestions contact:
% Elisavet Chatzilari ehatzi@iti.gr

dbstop if error

%% Initialize file and folder names
% Visual and textual kernels and annotations
kernelFolder = './kernels/';
VisualKernelFileName = [kernelFolder 'VisualKernels.mat'];
VisualAnnotationFileName = [kernelFolder 'VisualAnnotations.mat'];
TextualKernelFileName = [kernelFolder 'TextualKernels.mat'];
TextualAnnotationFileName = [kernelFolder 'TextualAnnotations.mat'];


featureFolder = './features/';
% create folder 
if ~exist(featureFolder,'dir')
    mkdir(featureFolder);
end

% maturity and reliability features - they will be calculated if they do
% not exist given kernels and annotations 
maturityFileName = [featureFolder 'Visual_APbyCV_15k_3fold.mat'];
reliabilityFileName = [featureFolder 'Textual_APbyCV_15k_3fold.mat'];

% Bootstrapping results; average precision of baseline and enhanced
% classifiers
InitialFileName = [featureFolder 'Results_Initial_15k.mat'];
if ~exist(InitialFileName,'file')
    disp('Provide the average precision for the baseline classifiers...');
    exit;
end
FinalFileName = [featureFolder 'Results_Final_15k.mat'];
if ~exist(FinalFileName,'file')
    disp('Provide the average precision for the enhanced classifiers...');
    exit;
end

%% Compute Features

% Visual - Maturity
if ~exist(maturityFileName,'file')
    if ~exist(VisualKernelFileName,'file')
        disp('Provide the visual kernel to proceed...');
        exit;
    end
    load(VisualKernelFileName, 'TrainKernel');
    if ~exist(VisualAnnotationFileName,'file')
        disp('Provide the annotations for the visual kernel...')
        exit;
    end
    load(VisualAnnotationFileName, 'anns');
    % calculates the maturity feature using 3-fold cross validation
    APmat = ComputeAPbyCV(TrainKernel,anns,3,maturityFileName);
else
    load(maturityFileName);
    APmat = AP;
end

% Textual - Reliability
if ~exist(reliabilityFileName,'file')
    if ~exist(VisualKernelFileName,'file')
        disp('Provide the textual kernel to proceed...');
        exit;
    end
    load(TextualKernelFileName, 'TrainKernel');
    if ~exist(TextualAnnotationFileName,'file')
        disp('Provide the annotations for the textual kernel...')
        exit;
    end
    load(TextualAnnotationFileName, 'anns');
    % calculates the reliability feature using 3-fold cross validation
    APrel = ComputeAPbyCV(TrainKernel,anns,3,reliabilityFileName);
else
    load(reliabilityFileName);
    APrel = AP;
end

% compute the mean average precision over the N folds
APmat = mean(APmat,2);
APrel = mean(APrel,2);
X = [APmat APrel];

%% Compute Output Values

% compute the performance of the initial and final models based on your
% chosen bootstrapping method
In = load(InitialFileName, 'AP');
Fin = load(FinalFileName, 'AP');
y = Fin.AP-In.AP;

%% Train & Test Regression Model

% trains the regression model and computes the leave-one-out estimates
[ preds SME ] = Generalizing2Concepts( X, y );

%% Plot results

N = numel(y);

% Random ranking
rng(13);
p = randperm(N);
randOrder = y(p);
% Proposed ranking using the regression results
[~,p] = sort(preds,'descend');
propOrder = y(p);
% Best case scenario ranking
[~,p] = sort(y,'descend');
BestOrder = y(p);
% Worst case scenario ranking
[~,p] = sort(y,'ascend');
WorstOrder = y(p);

figure;
hold on
plot(cumsum(randOrder),'k');
plot(cumsum(propOrder),'m');
plot(cumsum(BestOrder),'r');
plot(cumsum(WorstOrder),'b');

legend('Random','Proposed approach','Upper baseline','Lower baseline');

end
