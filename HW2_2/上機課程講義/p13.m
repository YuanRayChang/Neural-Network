%=========================================================================%
% NN Toolbox                                                              %
% Example 1                                                               %
% FNN                                                                     %
% 08.11.17  no.3                                                          %
%=========================================================================%
clc;
clear all;
close all;
% Collect data
P = [0 -1 2 -3 4 -5 6 -7 8 -9 10];
T = [0 1 2 3 4 5 6 7 8 9 10];
Pt = [0 1 -2 3 -4 5 -6 7 -8 9 -10];
Tt = [0 1 2 3 4 5 6 7 8 9 10];

% Create network
net = newff(P,T,10);
% net

%=========================================================================%
% Set training parameter values: net.trainParam                           %
%=========================================================================%
net.trainParam.epochs = 100;
net.trainParam.show = 25;
net.trainParam.lr = 0.01;
net.trainParam.goal = 0;
%=========================================================================%
% Train network
net = train(net, P, T,[],[],[],[]);

% Test network
Tt
yt = sim(net,Pt)

% Error
error_t = mse(yt-Tt)
