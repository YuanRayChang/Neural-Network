%=========================================================================%
% NN Toolbox                                                              %
% Example 1                                                               %
% FNN                                                                     %
%                                                         %
%=========================================================================%
clc;
clear all;
close all;
% Collect data
input=  load('bezdekIris.data1.txt');
for i=1:4
    input(:,i)=(input(:,i)-min(input(:,i)))./(max(input(:,i))-min(input(:,i)));
end
P(1:30,1:4)=input(1:30,1:4);
P(31:60,1:4)=input(51:80,1:4);
P(61:90,1:4)=input(101:130,1:4);
T(1:30)=input(1:30,5);
T(31:60)=input(51:80,5);
T(61:90)=input(101:130,5);


%Pt = [0 1 -2 3 -4 5 -6 7 -8 9 -10];
%Tt = [0 1 2 3 4 5 6 7 8 9 10];
Pt(1:20,1:4)=input(31:50,1:4);
Pt(21:40,1:4)=input(81:100,1:4);
Pt(41:60,1:4)=input(131:150,1:4);
Tt(1:20)=input(31:50,5);
Tt(21:40)=input(81:100,5);
Tt(41:60)=input(131:150,5);
%%
% Create network
% net = newff(P',T,[5 5],{'tansig','tansig','purelin'});
%net = newff(P',T,[5 5],{'logsig','logsig','purelin'});
net = newff(P',T,[5 5],{'tansig','logsig','purelin'},'trainscg'); %Scaled Conjuate Gradient
%net = newff(P',T,[5 5],{'tansig','logsig','purelin'},'trainbfg'); %Quasi_Newton
%net = newff(P',T,[5 5],{'tansig','logsig','purelin'},'trainbfg');  %Scaled Conjuate Gradient with powell

% net

%=========================================================================%
% Set training parameter values: net.trainParam                           %
%=========================================================================%
net.trainParam.epochs = 400;
net.trainParam.show = 25;
net.trainParam.lr = 0.0015;
net.trainParam.goal = 0;
%=========================================================================%
% Train network
net = train(net, P', T);

% Test network
yt = sim(net,Pt');
plot(yt,'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count=0;
for i=1:60
    if yt(i)<1.5
    yt(i)=1;
    elseif yt(i)<2.5  && 1.5<yt(i)
    yt(i)=2;
    elseif 2.5<yt(i)
    yt(i)=3;
    end
ERR(i)=Tt(i)-yt(i);
    if ERR(i)==0
    count=count+1;
    end

end

accuracy=count/60*100

figure(2)

plot(1:60,Tt);
hold on;
plot(1:60,yt,'ro');
xlabel('sample');
ylabel('sort');
title(['test accuracy=',num2str(accuracy),'%'])
% Error
error_t = mse(yt-Tt);