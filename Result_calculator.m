disp('BRBaDE_best');
%pred_addr = strcat('Predictions\','BRBaDE_best_balanced','_pred.mat');
pred_addr = strcat('Predictions\','BRBaDE_best_balanced_3','_pred.mat');
%pred_addr = strcat('Predictions\','BRBaDE_r1','_pred.mat');
%pred_addr = strcat('Predictions\','BRBaDE_r1_v2','_pred.mat');
dataset = readmatrix('Data\Dataset.csv');
load('5-fold_indices.mat','indices');
load(pred_addr);
results = [];
p = [];
t = [];
for k = 1 : 5
    ts = dataset(test(indices,k),:);
    test_targ = ts(:,end);
    eval(['pred = ' 'pred_' num2str(k) ';']);
    p = [p pred];
    t = [t test_targ];
    [acc, prec, reca, spec, f1_score] = result_generator(pred,test_targ);
    results = [results ; [acc, prec, reca, spec, f1_score]];
    writematrix(pred,strcat('Predictions\','Pred_',num2str(k),'.csv'));
end 

titles = ["acc","prec","reca","spec","f1_score"];
results

function [acc, prec, reca, spec, f1_score] = result_generator(pred,Target)
%pred = (Pred <= 0.25)*0 + (Pred > 0.25 & Pred <= 0.75) * 1 + (Pred > 0.75) * 2;
%pred = (Pred > 0.4383);
% MSE = mse(Target, Pred);
acc = [];
prec = [];
reca = [];
spec = [];
f1_score = [];
class = [0 1 2];
for c = 0 : 2
    pos = [c];
    neg = class(class~=c);
    cp = classperf(Target,pred,'Positive',pos,'Negative',neg);
    acc = [acc cp.CorrectRate];
    prec = [prec cp.PositivePredictiveValue];
    reca = [reca cp.Sensitivity];
    spec =  [spec cp.Specificity];
    f1_score = [f1_score (2 * (cp.PositivePredictiveValue * cp.Sensitivity) / (cp.PositivePredictiveValue + cp.Sensitivity))];
end    
% acc = mean(acc,'all');
% prec = mean(prec,'all');
% reca = mean(reca,'all');
% spec = mean(spec,'all');
% f1_score= mean(f1_score,'all');
%[X,Y,T,AUC] = perfcurve(Target,pred,1);
end
