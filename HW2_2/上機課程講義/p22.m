%==============================%
% NN Toolbox                                              %
% Use FNN to classify digitas                     %
% 2010.10.14                                                %
%==============================%
clc;
close all;
clear all;
load exerciseTwo;

%------------------------------------------------------%
% FNN                                                          %
%------------------------------------------------------%
ffnet = newff(P, T, 20, {'logsig', 'tansig'},'trainrp');

ffnet.trainParam.epochs = 1000;
ffnet.trainParam.lr = 0.01;
ffnet.trainParam.goal = 0;

ffnet = train(ffnet, P, T, [], [], [], []);

y = sim(ffnet, test_letter)
y = compet(y);
test_index
Y = vec2ind(y)
