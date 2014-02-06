function [AP] = ComputeAPbyCV(K,anns,N,fileName)
% ComputeAPbyCV: computes the features for the regression model
% K: kernel (visual or textual)
% anns: annotations
% N: number of folds
% fileName: the name of the file where the results are to be stored

% number of concepts
Nci = size(anns,2);
K = double(K);

for ci=1:Nci
    disp(['***Computing AP for concept ' mat2str(ci)]);
    labels = anns(:,ci);
    c = cvpartition(labels,'kfold',N);
    % do CV
    for i=1:N
        idx = training(c,i);
        model = svmtrain(labels(idx), [(1:sum(idx))' K(idx,idx)], '-t 4 -c 2.2');
        [~,~,s] = svmpredict(labels(~idx), [(1:sum(~idx))', K(~idx,idx)], model);
        if model.Label(1) == -1
            s = -s;
        end
        AP(ci,i) = averagePrecision(labels(~idx),s);
    end
end

save(fileName,'AP');