close all;clear all;clc;
P = [-1 0 1 1 0 -1; ...
1 0 1 -1 0 -1];
T = [0 1 1 0 -1 -1;0 1 -1 0 -1 1];
P_seq = con2seq(P);
T_seq = con2seq(T);


elmnet = newelm(P_seq, T_seq, 15, {'tansig','purelin'},'traingdm');
elmnet.trainParam.epochs = 1000;
elmnet.trainParam.lr = 0.1;
elmnet.trainParam.goal = 0;
elmnet = train(elmnet, P_seq, T_seq, [], [], [], []);
Y_seq = sim(elmnet,P_seq);
Y = cell2mat(Y_seq);

figure(1)
S = [0 1 1 0 -1 -1 0;0 1 -1 0 -1 1 0];
plot(S(1,:),S(2,:),'b-o','LineWidth',3);
axis([-1.5 1.5 -1.5 1.5]);
hold on;
Y1(1,:) = [Y(1,:) Y(1,1)];
Y1(2,:) = [Y(2,:) Y(2,1)];
plot(Y1(1,:),Y1(2,:),'r-o','LineWidth',3);