%A BRBaDE that optimizes the input referential variables too
%Load dataset
dataset = readmatrix('Data\Dataset.csv');
load('5-fold_indices.mat','indices');
scaling_lowerbound = min(dataset(:,1:end-1));
scaling_upperbound = max(dataset(:,1:end-1));
features = dataset(:,1:end-1);
features = rescale(features, 'InputMin', scaling_lowerbound ,'InputMax', scaling_upperbound);
dataset(:,1:end-1) = features;
clear features;

%Determine bounds for utility and data size
sizeData = size(dataset(:,1:end-1));
upperBound = ones(1,sizeData(2));
lowerBound = zeros(1,sizeData(2));
midPoint = (upperBound + lowerBound) / 2; %This line is basically useless, but I still kept it for some complex reason. Don't remove it.

utilities = [lowerBound' midPoint' upperBound'];
nARef = 3; %number of referential values
nCRef = 3;
nAttr = sizeData(2);
nRules = nARef;

parpool('local',6);
rng shuffle


%d = nARef*nRules + nRules + nARef
d = (nARef - 2) + nCRef*nRules + nRules + nCRef; % d represents number of variables. 
                                             % Number of Beta_jk parameters
                                             % = nARef*nRules
                                             % Number of theta parameters
                                             % = nRules
                                             % Number of consequent utility 
                                             % values to be optimized 
                                             % = Number of consequent 
                                             % reference values. 
%The following values of F                                             
F =  0.7 + rand * (0.9 - 0.7); %DE parameter - Scaling (0.5 to 0.9)
CR = 0.5; %DE parameter - Crossover probabilty
NP = 10*d ;%Initial Population Size
Lb = zeros(1,d);
Ub = ones(1,d);
for k =  1 : 5
    tr = dataset(training(indices,k),:);
    train_feat = tr(:,1:end-1);
    sizeTrain= size(tr);
    nTrain = sizeTrain(1);
    train_targ = tr(:,end);
    num_zeros = sum(train_targ(:)==0);
    num_ones = sum(train_targ(:)==1);
    num_twos = sum(train_targ(:)==2);
    class_proportions = [num_zeros num_ones num_twos];
    [elems,class_weights] = sort(class_proportions);
    [best, fmin] = BRBaDE(d,F,CR,NP,Lb,Ub,train_feat,train_targ,utilities,class_weights,nAttr,nARef,nCRef,nRules,nTrain,k);    
    
    eval(['model_' num2str(k) '= best;' ]);
    save('Models\BRBaDE_best_balanced_3.mat',strcat('model_',num2str(k)),'-append');        
end    



delete(gcp('nocreate'));

function [best, fmin] = BRBaDE(d, F, CR, NP, Lb, Ub, train_feat,train_targ,utilities,class_weights,nAttr,nARef,nCRef,nRules,nTrain,k)

tol = 10 ^ (-6); %Stop Tolerance
N_iter = 5000; %Number of itertions
%Initiate the solution/population
G = 1;
g = 1;
parfor i = 1:NP
    
    f = Lb + (Ub - Lb) .* rand(size(Lb));
    Sol(i,:,g) = f;%[util f]; %For a 3 reference value system utilities(:2:end - 1) is simply 2.
    [Fitness(g,i), Sol(i,:,g)] = FitnessFunction(Sol(i,:,g),train_feat,train_targ,utilities,class_weights,nAttr,nARef,nCRef,nRules,nTrain);
end    


%Calculate Current best
[fmin,I] = min(Fitness(g,:));
best = Sol(I,:,g);
%Start the iteration by DE
r1 = zeros(NP,d);
r2 = zeros(NP,d);
r3 = zeros(NP,d);
v = zeros(NP,d);
Fnew = zeros(N_iter,NP);
while ((fmin>tol) && (N_iter >= G))
    
    if (G >= 2)
        
        g = 2; 
        err_sqr = (Sol(:,:,g) - Sol(:,:,g - 1)).^2 ;
        PC = sqrt((sum(sum(err_sqr,2),1))./NP);
        fit_sqr = (Fitness(g,:) - Fitness(g - 1,:)) .^ 2; 
        FC = sqrt((sum(fit_sqr,2)) ./ NP);
        d11 = 1 - ((1 + PC).*exp(-PC));
        d12 = 1 - ((1 + FC).*exp(-FC));
%         d21 = 2 .* d11;
%         d22 = 2 .* d12; 
        d21 = d11;
        d22 =  d12;         
        [CR,F] = brbes(d11,d12,d21,d22);    
        Sol(:,:,g-1) = Sol(:,:,g);
        Fitness(g-1,:) = Fitness(g,:);
        g = 1;
        
    end 

    % donor vector creation
    % In the following code segment starting with 'Donor Start' and 'Donor
    % End' Comment, we will be creating the donor vectors for each solution
    % in the population without using any loops i.e. vectorization
    
    % Donor Start
    % Step - 1: Create NP row vectors of r1, r2 and r3
    % The values r1,r2 and r3 of the ith row vector need to be such that i not in (r1,r2,r3).
    % To implement this, we first need a matrix of size (NP,NP-1) where each row contains the numbers
    % 1:NP such that the the i-th row does not contain number i. 
    % I noticed that a matrix of size (NP,NP-1) containing numbers 1:NP-1
    % in each row can be converted into the matrix mentioned above by
    % simply adding it to a modified form of the compliment of a lower triangular 
    % matrix (N.B. - Compliment of lower triangular matrix is not an upper triangular matrix).
    % Let us consider an example where NP = 6. The matrix we want looks as
    % follows
    % P  = [2 3 4 5 6;
    %       1 3 4 5 6;
    %       1 2 4 5 6;
    %       1 2 3 5 6;
    %       1 2 3 4 6;
    %       1 2 3 4 5];
    % Initial Matrix,
    % P0  = [1 2 3 4 5;
    %       1 2 3 4 5;
    %       1 2 3 4 5;
    %       1 2 3 4 5;
    %       1 2 3 4 5;
    %       1 2 3 4 5];
    % The Compliment of a Lower triangular matrix of size (5,5)
    % I = [0 1 1 1 1;
    %      0 0 1 1 1;
    %      0 0 0 1 1;
    %      0 0 0 0 1;
    %      0 0 0 0 0];
    % I is to be modified by adding a ones row at the top of length. Thus I
    % becomes
    % I = [1 1 1 1 1;
    %      0 1 1 1 1;    
    %      0 0 1 1 1;
    %      0 0 0 1 1;
    %      0 0 0 0 1;
    %      0 0 0 0 0];
    % Adding I with P0 will give us P
    Devoid_i = repmat(1:NP-1,NP,1); % E.g, for NP = 3, P = [ 1 2; 1 2; 1 2];
    clt = ~tril(ones(NP-1,NP-1)); % tril form lower triangular matrix.
    mod_clt = [ones(1,NP-1);clt];
    Devoid_i = Devoid_i + mod_clt;

    % Next we create a (NP,3) matrix where each row consists three random
    % numbers picked from 1:NP-1. These are indices as per which the values
    % r1,r2 and r3 will be later selected from each row of Devoid_i
    [~, out] = sort(rand(NP,NP-1),2); %out contains the indices of the sorted random numbers
    x = out(:,1:3); %Selecting just 1st 3 columns. Any three columns can be taken. Randomness will prevail all the same
    col_ind = reshape(x',1,3*NP); %Reshaping for linear indexing. From each row of Devoid_i, 3 numbers are needed as 
                                 %r1, r2 and r3. col_ind represents the
                                 %column numbers of Devoid_i that contain
                                 %the r1s, r2s and r3s.
    row_ind = repelem(1:NP,1,3); % Like col_ind, row_ind represents the row indices of devoid_i's selected elements
    ind = sub2ind(size(Devoid_i),row_ind,col_ind); % row and column information turned into linear indices
    pqr = Devoid_i(ind); %Linear array containing r1s, r2s and r3s. E.g, r11,r21,r31,r12,r22,r32.....r1NP,r2NP,r3NP
    pqr = (reshape(pqr,3,NP))'; %Reshaping followed by transposing since matlab reshapes column-first
    
    %Step-2: Creat the r1, r2 and r3 matrices
    r1(:,:) = Sol(pqr(:,1),:,g);   
    r2(:,:) = Sol(pqr(:,2),:,g);
    r3(:,:) = Sol(pqr(:,3),:,g);

    
    %Step-3: Create Donor vector
    v(:,:) = r1(:,:) + F .* (r2(:,:)- r3(:,:)); %Mutated Donor Vector

    %v(:,:) = best + F .* (r2(:,:)- r3(:,:)); %Mutated Donor Vector
    
    %Donor End
    %Cross-over matrix
    Jr = randi(d,[NP,1]) == repmat(1:d,NP,1); %Cross-Over position
    K = ((rand(NP,d) < CR) | Jr); %Crossover filter
    efficient_sol(:,:) = Sol(:,:,g); %Shrinking dimensions to prevent broadcasting in the coming parfor
    v(:,:) = efficient_sol(:,:) .* (1 - K) + v(:,:) .* K;%Trial Vector Population
    %proxy(:,:) = v(:,:,G+1); %proxy is used to prevent parfor memory problem due to inconsistent indexing
    parfor m = 1 : NP           
        [fnew(m),v(m ,:)] = FitnessFunction(v(m,:),train_feat,train_targ,utilities,class_weights,nAttr,nARef,nCRef,nRules,nTrain);
    end
    
    Fnew(g+1,:) = fnew;
    fit = (Fnew(g,:) <= Fitness(g,:));
    Sol(:,:,g+1) = repmat((1 - fit)',1,d) .* Sol(:,:,g) + repmat(fit',1,d) .* v(:,:) ;
    Fitness(g+1,:) = (1 - fit) .* Fitness(g,:) + fit .* Fnew(g+1,:);
    %Update Current best
    [fminNew,Inew] = min(Fitness(g+1,:));
    if fminNew <= fmin
        best = Sol(Inew,:,g+1);
        fmin = fminNew;
        eval(['CR_' num2str(k) '= CR;']);
        eval(['F_' num2str(k) '= F;']);
        if k == 1
            save('Models\BRBaDE_best_balanced_3.mat',strcat('CR_',num2str(k)),strcat('F_',num2str(k)));        
        else 
            save('Models\BRBaDE_best_balanced_3.mat',strcat('CR_',num2str(k)),strcat('F_',num2str(k)),'-append');        
        end            
    end       

    %Output/display
    disp(['Number of evaluations : ',num2str(G)]);
    disp(['k : ',num2str(k)]);
    disp(['fmin : ',num2str(fmin)]);
    disp(['fminNew : ',num2str(fminNew)]);
    disp(['CR : ',num2str(CR)]);
    disp(['F : ',num2str(F)]);  
  
%     disp('Fitness(1)')
%     Fitness(1,1:6)
%     disp('Fitness(2)')
%     Fitness(2,1:6)
%     if G >=2
%         disp(['PC : ',num2str(PC)]);  
%         disp(['FC : ',num2str(FC)]); 
%         disp(['d11 : ',num2str(d11)]);
%         disp(['d12 : ',num2str(d12)]);
%         Ti
%     end       

    G = G + 1;
%     disp(['Sum of Beta_1 : ',num2str(sum(beta_1,2))]); 
%     disp(['Beta_1 : ',num2str(beta_2)]); 
%     disp(['Sum of Beta_2 : ',num2str(sum(beta_2,2))]); 

    
%     disp(['CR: ',num2str(CR)]);
%     disp(['F = ',num2str(F)]);
    
end
end
function [Cr,f] = brbes(D11,D12,D21,D22)
%Input transform
% The input transformation of the two BRBs is done together
in = [D11;D12;D21;D22];
h = [0 0.5 1; %preference degrees for D11 (Crossover antecedent)
     0 0.5 1; %preference degrees for D12 (Crossover antecedent)
     0 0.5 1; %preference degrees for D21 (Crossover antecedent)
     0 0.5 1]; %preference degrees for D22 (Crossover antecedent) 
    
% twos = 2.* ones(2,1);
% one_n_two = [ones(2,1);twos];
h_den = [ones(4,1) diff(h,1,2)]; 
%h_den = [one_n_two diff(h,1,2)];
lp = max(0,(h - in)); %piecewise linear proportion
lp = lp .* (lp < h_den);
alpha = lp ./ h_den;
alpha = circshift(alpha,-1,2);
res_ass =  1 - alpha;
res_ass = res_ass .* (alpha~=0);
res_ass = circshift(res_ass,1,2);
Ti = alpha + res_ass;
Ti = Ti + (h == in);
theta_1 = rand(9,1);
del_1 = rand(9,2);
del_bar_1 = del_1 ./ max(del_1,[],2);
theta_2 = rand(9,1);
del_2 = rand(9,2);
del_bar_2 = del_2 ./ max(del_2,[],2);
match_1 = [Ti(1,1)  Ti(2,1); 
           Ti(1,1)  Ti(2,2);
           Ti(1,1)  Ti(2,3);
           Ti(1,2)  Ti(2,1); 
           Ti(1,2)  Ti(2,2);
           Ti(1,2)  Ti(2,3);
           Ti(1,3)  Ti(2,1); 
           Ti(1,3)  Ti(2,2);
           Ti(1,3)  Ti(2,3)];
match_1 = match_1 .^  del_bar_1;   
combined_match_1 = prod(match_1,2);
rule_weight_product_1 = combined_match_1 .* theta_1;
w_1 = rule_weight_product_1 ./ sum(rule_weight_product_1,1);
match_2 = [Ti(3,1)  Ti(4,1); 
           Ti(3,1)  Ti(4,2);
           Ti(3,1)  Ti(4,3);
           Ti(3,2)  Ti(4,1); 
           Ti(3,2)  Ti(4,2);
           Ti(3,2)  Ti(4,3);
           Ti(3,3)  Ti(4,1); 
           Ti(3,3)  Ti(4,2);
           Ti(3,3)  Ti(4,3)];     
match_2 = match_2 .^  del_bar_2;
combined_match_2 = prod(match_2,2);
rule_weight_product_2 = combined_match_2 .* theta_2;
w_2 = rule_weight_product_2 ./ sum(rule_weight_product_2,1);

b1 = rand(3,9);
sum_of_beliefs = sum(b1,1);
if sum_of_beliefs > 0
    indices_of_sums_greater_than_1 = sum_of_beliefs > 1;
    difference_than_1 = (sum_of_beliefs - ones(1,9)) .* indices_of_sums_greater_than_1 ;
    distribution_of_difference = (b1 ./ sum_of_beliefs).* difference_than_1;
    b1 = (b1 - distribution_of_difference)';
end
Ti1 = Ti(1:2,:);
b1 = b1 .* (sum(sum(Ti1,2),1)./2);

b2 = rand(3,9);
sum_of_beliefs = sum(b2,1);
if sum_of_beliefs > 0
    indices_of_sums_greater_than_1 = sum_of_beliefs > 1;
    difference_than_1 = (sum_of_beliefs - ones(1,9)) .* indices_of_sums_greater_than_1 ;
    distribution_of_difference = (b2 ./ sum_of_beliefs).* difference_than_1;
    b2 = (b2 - distribution_of_difference)';
end    
Ti2 = Ti(3:4,:);
b2 = b2 .* (sum(sum(Ti2,2),1)./2);
m_jk_1 = w_1 .* b1;
m_jk_2 = w_2 .* b2;
%m_Dk = 1 - sum(m_jk,1);
m_bar_Dk_1 = 1 - w_1;
m_bar_Dk_2 = 1 - w_2;
m_tild_Dk_1 = w_1.*(1-sum(b1,2));
m_tild_Dk_2 = w_2.*(1-sum(b2,2));
k1 =  1 / (sum(prod(m_jk_1 + m_bar_Dk_1 + m_tild_Dk_1,1),2) - 2.* prod(m_bar_Dk_1 + m_tild_Dk_1, 1));
k2 =  1 / (sum(prod(m_jk_2 + m_bar_Dk_2 + m_tild_Dk_2,1),2) - 2.* prod(m_bar_Dk_2 + m_tild_Dk_2, 1));
Beta_1 = (k1.* ( prod(m_jk_1 + m_bar_Dk_1 + m_tild_Dk_1,1) - prod(m_bar_Dk_1 + m_tild_Dk_1, 1))) ./ (1 - (k1.* prod(m_bar_Dk_1,1)));
Beta_2 = (k2.* ( prod(m_jk_2 + m_bar_Dk_2 + m_tild_Dk_2,1) - prod(m_bar_Dk_2 + m_tild_Dk_2, 1))) ./ (1 - (k2.* prod(m_bar_Dk_2,1)));
Cr = 0.1*Beta_1(1) + 0.45*Beta_1(2)+0.8*Beta_1(3); %These values were selected in light of the book nature inspired algorithm
f = 0.4*Beta_2(1) + 0.675*Beta_2(2)+ 0.95*Beta_2(3); %These values were selected in light of the book nature inspired algorithm
end


function [fit, sol]= FitnessFunction(x,train_feat,train_targets,utilities,class_weights,nAttr,nARef,nCRef,nRules,nTrain)
%global nAttr nARef nTrain nRules train_features output_utilities train_targets;   
ref_param_range = nARef - 2;
beta_jk_param_range = ref_param_range + nCRef * nRules;
theta_jk_param_range = beta_jk_param_range + nRules;
cons_utility_param_range = theta_jk_param_range + nCRef;

ref_params = x(:,1:ref_param_range); %First (nARef - 2) * nAttr are for non terminal utilities
beta_jk_params = x(:, (ref_param_range + 1):beta_jk_param_range); %2nd nARef*nRules are for beta_jk
theta_k_params = x(:,(beta_jk_param_range + 1):theta_jk_param_range);
cons_utility_params = x(:,(theta_jk_param_range + 1):cons_utility_param_range);
%utility_params = x(:,(nARef * nRules + nRules + 1):(nARef * nRules + nRules + nARef));


% ref_params = abs(ref_params);
% ref_params = (ref_params .* (ref_params < 1)) + ((ref_params > 1) .* ref_params - (ref_params > 1)) ;
rnd = rand(1,nARef - 2);
ref_params = ref_params .* (ref_params > 0 & ref_params < 1) + rnd .* (ref_params <= 0 | ref_params >= 1);
refs = repmat(ref_params,1,nAttr);
utilities(:,2:end-1) = sort(reshape(refs,nARef-2,nAttr),1)';
transformed_input = input_Transform(train_feat,utilities,nAttr,nARef,nTrain);

%beta_jk_params = (beta_jk_params >= 0) .* beta_jk_params .* (beta_jk_params <= 1) + (beta_jk_params > 1);
rnd = rand(1,nCRef * nRules);
beta_jk_params = beta_jk_params .* (beta_jk_params > 0 & beta_jk_params < 1) + rnd .* (beta_jk_params <= 0 | beta_jk_params >= 1);

% theta_k_params = abs(theta_k_params);
% theta_k_params = theta_k_params .* (theta_k_params <= 1) + (theta_k_params > 1);
rnd = rand(1,nRules);
theta_k_params = theta_k_params .* (theta_k_params >= 0 & theta_k_params <= 1) + rnd .* (theta_k_params < 0 | theta_k_params > 1);

%cons_utility_params = (cons_utility_params >= 0) .* cons_utility_params .* (cons_utility_params <= 1) + (cons_utility_params > 1);
rnd = rand (1,nCRef);
cons_utility_params =  cons_utility_params .* (cons_utility_params >= 0 & cons_utility_params <= 1) + rnd .* (cons_utility_params < 0 | cons_utility_params > 1);
%output_utilities = [0 33.33 66.67 100];
attribute_usage_vector = ones(nAttr,nTrain); %For complete attribute cases


%Normalize beta_jk parameters
belief_matrix = reshape(beta_jk_params,nCRef,nRules); 
sum_of_beliefs = sum(belief_matrix,1);
belief_matrix = belief_matrix ./ sum_of_beliefs;
% if sum_of_beliefs > 0
%     indices_of_sums_greater_than_1 = sum_of_beliefs > 1;
%     difference_than_1 = (sum_of_beliefs - ones(1,nRules)) .* indices_of_sums_greater_than_1 ;
%     distribution_of_difference = (belief_matrix ./ sum_of_beliefs) .* difference_than_1;
%     belief_matrix = belief_matrix - distribution_of_difference ;
% end    
sol = [ref_params reshape(belief_matrix,1,nCRef*nRules) theta_k_params sort(cons_utility_params,'descend')];
belief_matrix = belief_matrix';

w_k = rule_activation_weights(transformed_input,theta_k_params,nAttr,nARef,nTrain,nRules); %Rule Activation Weights
Beta_jk = belief_matrix_generator(belief_matrix,transformed_input,attribute_usage_vector,nAttr,nARef,nTrain);%Initial Belief Matrix with belief update
Beta = BRBReasoner(Beta_jk,w_k,nCRef);
U = Training_Aggregate_Class(Beta,cons_utility_params,class_weights,nTrain);
fit = sum((U ~= train_targets),'all') / nTrain;
% U = Training_Aggregate_Reg(Beta,cons_utility_params,nTrain);
% fit =  immse(U,train_targets); 
%Mean Squared Error


%fit = classerror(U,train_targets);
%fit = sqrt(fit);
%fit = MAE(U,train_targets);
end

% function train_aggregator_reg = Training_Aggregate_Reg(Beta,output_utilities,nTrain)
% %output_utilities = sort(output_utilities);
% weighing = Beta .* output_utilities;
% output = sum(weighing,2);
% train_aggregator_reg = reshape(output,nTrain,1);
% end

function train_aggregator_class = Training_Aggregate_Class(Beta,output_utilities,class_weights,nTrain)
output_utilities = sort(output_utilities,'descend');
[useless,original_order] = sort(class_weights);
Beta = Beta(:,class_weights,:).* output_utilities; %As our classes are 0,1 and 2. So we scale [0,1] output utilities
Beta = Beta(:,original_order,:);
[mx,idx] = max(Beta,[],2);
train_aggregator_class = reshape(idx - 1,nTrain,1); %Since main dataset classes are 0,1 and 2.
end

function brb = BRBReasoner(Beta_jk,w_k,nCRef)
%global nARef;
m_jk = w_k.*Beta_jk;
%m_Dk = 1 - sum(m_jk,1);
m_bar_Dk = 1 - w_k;
m_tild_Dk = w_k.*(1-sum(Beta_jk,2));
k =  1 / (sum(prod(m_jk + m_bar_Dk + m_tild_Dk,1),2) - (nCRef-1).* prod(m_bar_Dk + m_tild_Dk, 1));
Beta_j = (k.* ( prod(m_jk + m_bar_Dk + m_tild_Dk,1) - prod(m_bar_Dk + m_tild_Dk, 1))) ./ (1 - (k.* prod(m_bar_Dk,1)));
brb = Beta_j;
end

function bel_mat_gen = belief_matrix_generator(belief_matrix,trans_inp,auv,nAttr,nARef,nTrain)
%  b = rand(nARef,n_weights);
%  bmg  = (b./sum(b,1));
%global nAttr nARef nTrain;
A = repelem(auv,1,nARef);
A = A .* trans_inp;
A = reshape(A,nAttr,nARef,nTrain);
A = sum(A,[1 2]);
Attribute_Usage_sum = sum(auv,1);
Attribute_Usage_sum = reshape(Attribute_Usage_sum,1,1,nTrain);
A = A ./ Attribute_Usage_sum;
bel_mat_gen = belief_matrix .* A;

end


function mf = rule_activation_weights(trans_in,theta,nAttr,nARef,nTrain,nRules)

%global nAttr nARef nTrain nRules conjunctive_rule_combo del theta;
theta = theta';
A = reshape(trans_in,nAttr,nARef,nTrain);
wighted_alpha_k = A;
alpha_k  = sum(wighted_alpha_k,1);
alpha_k = reshape(alpha_k,nRules,1,nTrain);
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






