clc;
clear all;
close all;
load HRV_parameter.mat; 
select = [2, 7, 8, 12, 13, 18];
x = cell2mat([ 
    Asthma_para(2:66,select);
    Asthma_para(2:66,select);
    Asthma_para(2:66,select);
    Asthma_para(2:66,select);
    Normal_para(2:66,select);
    Normal_para(2:66,select);
    Normal_para(2:66,select);
    Normal_para(2:66,select);
    ]);
% input
for i=1:length(select)
    x(:,i)= (x(:,i)-mean(x(:,i)))./std(x(:,i));
end

% input preprocedure
to_dimens = length(select);
x_n = x;

% output
y = [ ones(65*4,1); zeros(65*4,1)];

%random and split train ans valid
train_num = 104*4;
test_num = 26*4;

rand_ind= randperm(130*4);
x_train = zeros(train_num, to_dimens);
y_train = zeros(train_num, 1);
x_test = zeros(test_num, to_dimens);
y_test = zeros(test_num, 1);

for i = 1:train_num
    x_train(i, :) = x_n(rand_ind(i), :);
    y_train(i, :) = y(rand_ind(i), :);
end
for i = 1:test_num
    x_test(i, :) = x_n(rand_ind(train_num+i), :);
    y_test(i, :) = y(rand_ind(train_num+i), :);
end

% SVM
SVMModel = fitcsvm(x_train,y_train, 'Standardize',false, 'KernelFunction','rbf', 'KernelScale','auto');
CVSVMModel = crossval(SVMModel);
% k-fold valid
classLoss = kfoldLoss(CVSVMModel);

result = predict(SVMModel,x_test);
test_result = result == y_test;

train_result = predict(SVMModel,x_train);
train_result = train_result == y_train;

succ = 0;
for i = 1:length(train_result)
    if train_result(i) == 1
        succ = succ+1;
    end
end
succ = succ/train_num;

succ_te = 0;
for i = 1:length(test_result)
    if test_result(i) == 1
        succ_te = succ_te+1;
    end
end
succ_te = succ_te/test_num;


fprintf('kfold : %d , Train Rate : %d,  Valid : %d', classLoss, succ*100, succ_te);
save('SVM_model.mat', 'SVMModel');