clc
clear 
close all

%% Iris
load 'Iris.dat';
X_iris_train=[(Iris(1:35,1:4));(Iris(51:85,1:4));(Iris(101:135,1:4))];
Y_iris_train=[(Iris(1:35,5));(Iris(51:85,5));(Iris(101:135,5))];
X_iris_test=[(Iris(36:50,1:4));(Iris(86:100,1:4));(Iris(136:150,1:4))];
Y_iris_test=[(Iris(36:50,5));(Iris(86:100,5));(Iris(136:150,5))];

X_iris_train=normalization(X_iris_train);
X_iris_test=normalization(X_iris_test);

p = randperm(size(X_iris_train,1));
X_iris_train=X_iris_train(p,:);
Y_iris_train=Y_iris_train(p,:);

%net1 = newff(X_iris_train',Y_iris_train',[2 3],{'tansig','logsig','purelin'},'trainscg');
net1 = newff(X_iris_train',Y_iris_train',[2 3],{'tansig','logsig','purelin'},'trainrp');
net1.trainParam.epochs = 400;
net1.trainParam.show = 25;
net1.trainParam.lr = 0.0015;
net1.trainParam.goal = 0;

[net1,tr] = train(net1, X_iris_train', Y_iris_train');

% Test network
yt = sim(net1,X_iris_test');
plot(yt,'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count=0;
for i=1:length(yt)
    if yt(i)<1.5
    yt(i)=1;
    elseif yt(i)<2.5  && 1.5<yt(i)
    yt(i)=2;
    elseif 2.5<yt(i)
    yt(i)=3;
    end
ERR(i)=Y_iris_test(i)-yt(i);
    if ERR(i)==0
    count=count+1;
    end
end

accuracy=count/length(yt)*100;

figure(2)
plot(1:length(yt),Y_iris_test);
hold on;
plot(1:length(yt),yt,'ro');
xlabel('sample','fontsize',20);
ylabel('sort','fontsize',20);
title(['test accuracy=',num2str(accuracy),'%'],'fontsize',18)
% Error
error_t = mse(yt-Y_iris_test');

set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'iris_rp', 'pdf') %Save figure

figure(3)
plotperform(tr);
set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'iris_rp_per', 'pdf') %Save figure