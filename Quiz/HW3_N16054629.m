close all;clear all;clc;
p1 = rand(5000,1)*4-2;
for i=5001:9000
p2(i-5000) = 1.05 * sin(pi*i/45);
end
P=[p1 ;p2'];
P=[0;0;0;p1 ;p2'];
T=zeros(9000,1);
for i=3:9002
   T(i+1)=(T(i)*T(i-1)*T(i-2)*P(i-1)*(T(i-2)-1)+P(i))/(1+T(i-2)^2+T(i-1)^2);
end

P_seq = con2seq(P');
T_seq = con2seq(T');

elmnet = newelm(P_seq, T_seq, 5, {'tansig','purelin'},'traingdm');
elmnet.trainParam.epochs = 60;
elmnet.trainParam.lr = 0.05;
elmnet.trainParam.goal = 0;
elmnet = train(elmnet, P_seq, T_seq, [], [], [], []);
weighting= cell2mat(elmnet.lw)
bias= cell2mat(elmnet.b)
for i=0:249
P_test1(i+1)=sin(pi*i/25);
end
P_test2=ones(1,250);
for i=750:999
  P_test3(i-749)= 0.3*sin (pi*i/25)+0.1*sin(pi*i/32)+0.6*sin(pi*i/10);
end
P_test = [ 0 0 0 P_test1 P_test2 -P_test2 P_test3];
P_test=P_test';
T_test=zeros(1000,1);
for i=3:1002
   T_test(i+1)=(T_test(i)*T_test(i-1)*T_test(i-2)*P_test(i-1)*(T_test(i-2)-1)+P_test(i))/(1+T_test(i-2)^2+T_test(i-1)^2);
end

P_test_seq = con2seq(P_test');
Y_test_seq = sim(elmnet,P_test_seq);
Y = cell2mat(Y_test_seq);

figure(1)
X=1:1003;
plot(X,T_test,'b-');
hold on;
plot(X,Y,'r-');
mse=sqrt(sum((Y-T_test').^2))/length(X);
str=sprintf('mse=%d',mse);
title(str)
