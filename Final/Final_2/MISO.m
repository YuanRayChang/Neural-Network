close all; clear; clc;

%% Inputs & Outputs
% Inputs
P1 = zeros(1,11000);
P1(1:6000) = random('unif',-2,2,1,6000);
P1(6001:11000) = 1.05 * sin(pi * (6001:11000) / 45);

P2 = zeros(1,11000);
P2(1:6000) = random('unif',-2,2,1,6000);
P2(6001:11000) = sin(pi * (6001:11000) / 45);

P = [P1;P2];

% Outputs
x1 = zeros(1,11000);
x2 = zeros(1,11000);
for i = 1:10999
    x1(i+1) = 0.5 * ( ( x1(i) / (1 + x2(i)^2) ) + P1(i) );
    x2(i+1) = 0.5 * ( ( (x2(i) * x1(i)) / (1 + x2(i)^2) ) + P2(i) );    
end
T = x1 + x2;


% Translate Data
P_s = con2seq(P); T_s = con2seq(T); 

%% RNN
elmnet = newelm(P_s, T_s, 10 , {'tansig','purelin'},'traingdm');

elmnet.trainParam.epochs = 300;
elmnet.trainParam.lr = 0.01;
elmnet.trainParam.goal = 0;

elmnet = train(elmnet, P_s, T_s, [], [], [], []);

%% Test
% Test Input
Pt = zeros(1,1000);
Pt(1:250) = sin(pi * (1:250) /25);
Pt(251:500) = 1;
Pt(501:750) = -1;
Pt(751:1000) = 0.3 * sin(pi*(751:1000)/25) + 0.1 * sin(pi*(751:1000)/32) + 0.6 * sin(pi*(751:1000)/10);

% Desired
figure(1)

s1 = zeros(1,1000);
s2 = zeros(1,1000);
for i = 1:999
    s1(i+1) = 0.5 * ( ( s1(i) / (1 + s2(i)^2) ) + Pt(i) );
    s2(i+1) = 0.5 * ( ( (s2(i) * s1(i)) / (1 + s2(i)^2) ) + Pt(i) );    
end
S = s1 + s2;

plot((1:1000) , S);
hold on

% Test Output
Pt_s = con2seq(Pt);
Y_s = sim(elmnet , Pt_s);
Y = cell2mat(Y_s);
plot((1:1000) , Y);

legend({'Desired','Test Result'})

% MSE
mse = sum(((Y-S).^2).^0.5)/1000;

disp(['MSE : ' , num2str(mse)])