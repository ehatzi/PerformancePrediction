function [ preds SME ] = Generalizing2Concepts( X, y )
%GENERALIZING2CONCEPTS Summary of this function goes here
%   Detailed explanation goes here

N = numel(y);
% estimate Leave-one-out SME
preds = zeros(N,1);
for i=1:N
    trids = boolean(ones(N,1));
    trids(i)=0;
    model = svmtrain(y(trids), X(trids,:), '-s 3 -t 2 -q');
    [~,mets,preds(~trids)] = svmpredict(y(~trids), X(~trids,:), model,'-q');
    SME(i) = mets(2);
end
SME = SME';

% % estimate SME using 5 fold CV 
% load('cv_part.mat')
% preds_cv = zeros(N,1);
% for i=1:c.NumTestSets
%     t=training(c,i);
%     model = svmtrain(y(t), X(t,:), '-s 3 -t 2');
%     t=test(c,i);
%     [~,mets,preds_cv(t)] = svmpredict(y(t), X(t,:), model);
%     SMEcv(i) = mets(2);
% end
% SMEcv = SMEcv';


end

