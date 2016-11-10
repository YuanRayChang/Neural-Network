%=========================================================================%
% NN Toolbox                                                              %
% Example 1                                                               %
% FNN                                                                     %
% 08.11.17  no.1                                                          %
%=========================================================================%
clc;
clear all;
close all;
% Collect data
P = [0 -1 2 -3 4 -5 6 -7 8 -9 10];
T = [0 1 2 3 4 5 6 7 8 9 10];

% Create network
net = newff(P,T,10);
net

% Train network
net = train(net, P, T);

% Test network
y = sim(net,P)

% Error
error = mse(y-T)
%=========================================================================%
% Show the weights of the trained networks                                %
%=========================================================================%
net.IW
net.IW{1}
net.b
net.b{1}
net.LW
net.LW{2,1}
