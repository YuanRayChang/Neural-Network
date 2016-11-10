%=========================================================================%
% NN Toolbox                                                              %
% Example 2                                                               %
% PNN                                                                     %
% 08.11.17  no.1                                                          %
%=========================================================================%
clc;
close all;
clear all;
load exerciseTwo;

figure(1)
subplot(3,3,1)
plotchar(P(:,1))
subplot(3,3,2)
plotchar(P(:,27))
subplot(3,3,3)
plotchar(P(:,53))
subplot(3,3,4)
plotchar(P(:,2))
subplot(3,3,5)
plotchar(P(:,28))
subplot(3,3,6)
plotchar(P(:,54))
subplot(3,3,7)
plotchar(P(:,3))
subplot(3,3,8)
plotchar(P(:,29))
subplot(3,3,9)
plotchar(P(:,55))
%-------------------------------------------------------------------------%
% PNN                                                                     %
%-------------------------------------------------------------------------%
Tc = ind2vec(T_for_PNN);
pnnnet = newpnn(P, Tc);
Yc = sim(pnnnet, test_letter);
test_index
Y = vec2ind(Yc)

