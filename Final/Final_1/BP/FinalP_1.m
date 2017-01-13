clc;
clear all;
close all;
rate = 0;
% % �]�w�`�V�m�ӼƻPk��
train_n = 100;
k = 20;
num_v = 15;
% Ū��.mat��
load HRV_parameter.mat; 
% �s�@input�x�}
x = cell2mat([ Asthma_para(2:66,2:26); Normal_para(2:66,2:26)]);
% [pc,score,latent,tsquare] = princomp(x); 
% transMatrix = pc(:,1:num_v);
% x = x*transMatrix;
% input�Ѽƥ��W��
for i=1:25
    x(:,i)= (x(:,i)-mean(x(:,i)))./std(x(:,i));
end
% �s�@output�x�}
y = [ ones(65,1); -ones(65,1)];
% x = rand(130,3);
% y = exp(x(:,1).*x(:,2)) + pi.^x(:,3);
% �H�����ͥ�e���Ҳ�(x_CV,y_CV)�ΰ��]������(x_unknown,y_unknown)
%while rate < 75
ind_CV = [ sort(randperm(65,round(train_n/2))) 65+sort(randperm(65,round(train_n/2)))];
ind_unknown = setdiff(1:130,ind_CV);
A = [0.28	0.04	0.16	0.02	0.02	0.38	0.12	0.52	0.02	0.04	0.44	0.12	0.1	0.04	0.04	0.12	0.22	0	0	0	0	0.42	0.02	0	0];
Choice = [];
for i = 1:25
    if A(i) > 0.1
        Choice = [ Choice i];
    end
end
Choice = [ 1 6 7 11 12 17];
x_CV = x(ind_CV,Choice);
y_CV = y(ind_CV);
x_unknown = x(ind_unknown,Choice);
y_unknown = y(ind_unknown);
% �p���e���Ҳռ�
K = length(y_CV);
% ��index
n = 1:K;
% �ͦ�LOOCV��index
indices = crossvalind('Kfold',K,k);
% �ͦ������g����
net = newff(x_CV(1:(K-1),:)',y_CV(1:(K-1))', 6,{'tansig','satlins'},'trainrp');
% �]�p�V�m�Ѽ�
net.trainParam.epochs = 400;
net.trainParam.show = 25;
net.trainParam.lr = 0.00001;
net.trainParam.goal = 0;
%net.trainParam.max_fail = 500;
%net.trainParam.min_grad = 0;
net.divideFcn = 'divideind';
for i = 1:k
    % index�����নŪ�����L��, tr = trian set, te = test set
    te = (indices == i);
    nonte = ~te;
    a = n(nonte);
    val = sort(randperm(K-length(n(te)),round(length(a)*0.1)));
    net.divideParam.valInd = a(val);
    net.divideParam.trainInd = setxor(n(nonte),a(val));
    net.divideParam.testInd = n(te);
    % �N�V�m���ե��L�ȥN�J������
    % [net.divideParam.trainInd,net.divideParam.valInd,net.divideParam.testInd] = [K,ind_tr,ind_val,n(te)];
    % �V�mBJ4
    [net,tr] = train(net,x_CV',y_CV');
%     figure();
%     plotperform(tr)
%     hold on;
end
y_unknown_nn = sim(net,x_unknown');
m2 = 0;
for i = 1:length(y_unknown)
    if y_unknown_nn(i) >= 0
        y_unknown_result(i) = 1;
    else 
        y_unknown_result(i) = -1;
    end
    sse2(i) = (y_unknown_result(i) - y_unknown(i)).^2;
    if sse2(i) == 0
        m2 = m2+1;
    end
end
rate = m2/length(y_unknown)*100;
fprintf(' rate = %.2f%%\n',rate);
%end
o = 1:length(y_unknown);
figure();
plot(o,y_unknown,o,y_unknown_result,'-o','LineWidth',2);
%axis([0 length(y_unknown) 0.75 2.25]);
grid on;
figure();
plot(o,sse2,'LineWidth',2);
%axis([0 length(y_unknown) 0 1.25]);
grid on;
