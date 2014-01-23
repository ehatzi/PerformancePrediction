function PerformancePrediction
%% function description
% Computes classifier maturity and oracle reliability, trains a regression
% model, calculates leave-one-out estimates and plots the results.
% Based on paper "PERFORMANCE PREDICTION OF BOOTSTRAPPING FOR IMAGE
% CLASSIFICATION", Chatzilari et al.
% 
% Input - Calculate features from visual and textual kernel
% tranns: Training set annotations; matrix of dimensions #training
% instances x # concepts with '1's wherever an image is positive for a
% concept and -1 otherwise.
% VisualKernels.mat: file including the training kernel for calculating the
% maturity feature - based on visual features
% TextualKernels.mat: file including the training kernel for calculating
% the reliability feature - based on the textual features, i.e. oracle
% Results_Initial.mat & Results_Final.mat: files containing the average
% precision of the baseline and enhanced classifiers respectively. They
% include a vector AP of dimension #concepts x 1
%
% Download the data for a demo run from:
% 

dbstop if error


%% Compute Features
load('mirAnns.mat', 'tranns');

% Visual - Maturity
if ~exist('Visual_APbyCV_15k_3fold.mat','file')
    load('VisualKernels.mat', 'TrainKernel');
    APmat = ComputeAPbyCV(TrainKernel,tranns,3,'Visual_APbyCV_15k_3fold.mat');
else
    load('Visual_APbyCV_15k_3fold.mat');
    APmat = AP;
end

% Textual - Reliability

if ~exist('Textual_APbyCV_15k_3fold.mat','file')
    load('TextualKernels.mat', 'TrainKernel');
    APrel = ComputeAPbyCV(TrainKernel,tranns,3,'Textual_APbyCV_15k_3fold.mat');
else
    load('Textual_APbyCV_15k_3fold.mat');
    APrel = AP;
end

APmat = mean(APmat,2);
APrel = mean(APrel,2);
X = [APmat APrel];

%% Compute Output Values

% compute the performance of the initial and final models based on your
% chosen bootstrapping method
In = load('Results_Initial_15k.mat', 'AP');
Fin = load('Results_Final_15k.mat', 'AP');
y = Fin.AP-In.AP;

%% Train & Test Regression Model

% trains the regression model and computes the leave-one-out estimates
[ preds SME preds_cv SMEcv ] = Generalizing2Concepts( X, y );

%% Plot results

N = numel(y);

rng(13);
p = randperm(N);
randOrder = y(p);
[~,p] = sort(preds,'descend');
propOrder = y(p);
% [~,p] = sort(preds_cv,'descend');
% propOrder_cv = y(p);
[~,p] = sort(y,'descend');
BestOrder = y(p);
[~,p] = sort(y,'ascend');
WorstOrder = y(p);

figure;
hold on
plot(cumsum(randOrder),'k');
plot(cumsum(propOrder),'m');
% plot(cumsum(propOrder_cv),'c');
plot(cumsum(BestOrder),'r');
plot(cumsum(WorstOrder),'b');

end
