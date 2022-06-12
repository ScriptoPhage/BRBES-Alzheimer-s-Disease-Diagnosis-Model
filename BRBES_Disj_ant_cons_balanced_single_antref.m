%model_addr = strcat('Models\','BRBaDE_best_balanced','.mat'); 
%pred_addr = strcat('Predictions\','BRBaDE_best_balanced','_pred.mat');
model_addr = strcat('Models\','BRBaDE_best_balanced_3','.mat');
pred_addr = strcat('Predictions\','BRBaDE_best_balanced_3','_pred.mat');
load(model_addr);
dataset = readmatrix('Data\Dataset.csv');
load('5-fold_indices.mat','indices');
scaling_lowerbound = min(min(dataset(:,1:end-1)));
scaling_upperbound = max(max(dataset(:,1:end-1)));
features = dataset(:,1:end-1);
features = rescale(features, 'InputMin', scaling_lowerbound ,'InputMax', scaling_upperbound);
dataset(:,1:end-1) = features;
clear features;

sizeData = size(dataset(:,1:end-1));
nARef = 3;
nCRef = 3;
nAttr = sizeData(2);
nRules = nARef;

for k = 1:5
    ts = dataset(test(indices,k),:);
    test_feat = ts(:,1:end-1);
    sizeTest= size(ts);
    nTest = sizeTest(1);
    test_targ = ts(:,end);
    num_zeros = sum(test_targ(:)==0);
    num_ones = sum(test_targ(:)==1);
    num_twos = sum(test_targ(:)==2);
    class_proportions = [num_zeros num_ones num_twos];
    [elems,class_weights] = sort(class_proportions);
    
    
    eval(['model = ' 'model_' num2str(k) ';']);
    ref_param_range = (nARef - 2);
    beta_jk_param_range = ref_param_range + nARef * nRules;
    theta_jk_param_range = beta_jk_param_range + nRules;
    cons_utility_param_range = theta_jk_param_range + nCRef;
    
    ref_params = model(:,1:ref_param_range); %First (nARef - 2) * nAttr are for non terminal utilities
    beta = model(:, (ref_param_range + 1):beta_jk_param_range); %2nd nARef*nRules are for beta_jk
    theta = model(:,(beta_jk_param_range + 1):theta_jk_param_range);
    conseq = model(:,(theta_jk_param_range + 1):cons_utility_param_range);
    %util = x(:,(nARef * nRules + nRules + 1):(nARef * nRules + nRules + nARef));
    refs = repmat(ref_params,1,nAttr);
    utilities = [zeros(nAttr,1) sort(reshape(refs,nARef-2,nAttr),1)' ones(nAttr,1)];
    attribute_usage_vector = ones(nAttr,nTest); %For complete attribute cases
    transformed_input = input_Transform(test_feat,utilities,nAttr,nARef,nTest);
    w_k = rule_activation_weights(transformed_input,theta,nAttr,nARef,nTest,nRules); %Rule Activation Weights

    belief_matrix = reshape(beta,nARef,nRules);
    belief_matrix =  belief_matrix';
    Beta_jk = belief_matrix_generator(belief_matrix,transformed_input,attribute_usage_vector,nAttr,nARef,nTest);%Initial Belief Matrix with belief update
    output_distribution = BRBReasoner(Beta_jk,w_k,nCRef);
    Output = Training_Aggregate_Class(output_distribution,conseq,class_weights,nTest);
    %Output = Training_Aggregate_Reg(output_distribution,nTest);
    eval(['pred_' num2str(k) ' = Output;']);
    if k == 1
        save(pred_addr,strcat('pred_',num2str(k)));
    else 
        save(pred_addr,strcat('pred_',num2str(k)),'-append');
    end
end 



function train_aggregator_class = Training_Aggregate_Class(Beta,conseq,class_weights,nTest)
conseq = sort(conseq,'descend');
[useless,original_order] = sort(class_weights);
Beta = Beta(:,class_weights,:) .* conseq; %As our classes are 0,1 and 2. So we scale [0,1] output utilities
Beta = Beta(:,original_order,:);

[mx,idx] = max(Beta,[],2);
train_aggregator_class = reshape(idx - 1,nTest,1); %Since main dataset classes are 0,1 and 2.
end

% function train_aggregator = Training_Aggregate(Beta,output_utilities,nTest)
% weighing = Beta .* output_utilities;
% output = sum(weighing,2);
% train_aggregator = reshape(output,nTest,1);
% end

function brb = BRBReasoner(Beta_jk,w_k,nARef)
m_jk = w_k.*Beta_jk;
%m_Dk = 1 - sum(m_jk,1);
m_bar_Dk = 1 - w_k;
m_tild_Dk = w_k.*(1-sum(Beta_jk,2));
k =  1 / (sum(prod(m_jk + m_bar_Dk + m_tild_Dk,1),2) - (nARef-1).* prod(m_bar_Dk + m_tild_Dk, 1));
Beta_j = (k.* ( prod(m_jk + m_bar_Dk + m_tild_Dk,1) - prod(m_bar_Dk + m_tild_Dk, 1))) ./ (1 - (k.* prod(m_bar_Dk,1)));
brb = Beta_j;
end

function bel_mat_gen = belief_matrix_generator(initial_belief_matrix,trans_inp,aug,nAttr,nARef,nTest)

%  b = rand(nARef,n_weights);
%  bmg  = (b./sum(b,1));
%global nAttr nARef nTrain;
A = repelem(aug,1,nARef);
A = A .* trans_inp;
A = reshape(A,nAttr,nARef,nTest);
A = sum(A,[1 2]);
Attribute_Usage_sum = sum(aug,1);
Attribute_Usage_sum = reshape(Attribute_Usage_sum,1,1,nTest);
A = A ./ Attribute_Usage_sum;
bel_mat_gen = initial_belief_matrix .* A;

end


function mf = rule_activation_weights(trans_in,theta,nAttr,nARef,nTest,nRules)
%global nAttr nARef nTrain nRules conjunctive_rule_combo del theta;
theta = theta';
A = reshape(trans_in,nAttr,nARef,nTest);
alpha_k  = sum(A,1);
alpha_k = reshape(alpha_k,nRules,1,nTest);
combo_rule_weight_numerator = alpha_k .* theta;
combo_rule_weight_denominator = sum(combo_rule_weight_numerator,1);
weights = combo_rule_weight_numerator ./ combo_rule_weight_denominator;
mf = weights;
end

function transformed_input = input_Transform(test_features,utilities,nAttr,nARef,nSamples)
h_denominators = [ones(nAttr,1) diff(utilities,1,2)]; 
h = repmat(utilities,1,nSamples);
h_denominators = repmat(h_denominators,1,nSamples); 
% col_set = repelem(train_features',1,nARef);
col_set = repelem(test_features',1,nARef);
linprop = max(0,(h - col_set)); %piecewise linear proportion
linprop = linprop .* (linprop < h_denominators);
alpha = linprop ./ h_denominators;
alpha = circshift(alpha,-1,2);
residual_assignment =  1 - alpha;
residual_assignment = residual_assignment .* (alpha~=0);
residual_assignment = circshift(residual_assignment,1,2);
transformed_input  = alpha + residual_assignment;
transformed_input = transformed_input + (h == col_set);
end





