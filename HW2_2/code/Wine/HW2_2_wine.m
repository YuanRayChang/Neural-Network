clc
clear 
close all

%% wine
load 'wine.data.txt';
X_wine_train=[(wine_data(1:50,2:14));(wine_data(60:115,2:14));(wine_data(131:170,2:14))];
Y_wine_train=[(wine_data(1:50,1));(wine_data(60:115,1));(wine_data(131:170,1))];
X_wine_test=[(wine_data(51:59,2:14));(wine_data(116:130,2:14));(wine_data(171:178,2:14))];
Y_wine_test=[(wine_data(51:59,1));(wine_data(116:130,1));(wine_data(171:178,1))];

X_wine_train=normalization(X_wine_train);
X_wine_test=normalization(X_wine_test);

p = randperm(size(X_wine_train,1));
X_wine_train=X_wine_train(p,:);
Y_wine_train=Y_wine_train(p,:);

%net1 = newff(X_wine_train',Y_wine_train',[4 2],{'tansig','logsig','purelin'},'trainscg');
net1 = newff(X_wine_train',Y_wine_train',[4 2],{'tansig','logsig','purelin'},'trainrp');
net1.trainParam.epochs = 400;
net1.trainParam.show = 25;
net1.trainParam.lr = 0.0015;
net1.trainParam.goal = 0;

[net1,tr] = train(net1, X_wine_train', Y_wine_train');

% Test network
yt = sim(net1,X_wine_test');
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
ERR(i)=Y_wine_test(i)-yt(i);
    if ERR(i)==0
    count=count+1;
    end
end

accuracy=count/length(yt)*100;

figure(2)
plot(1:length(yt),Y_wine_test);
hold on;
plot(1:length(yt),yt,'ro');
xlabel('sample','fontsize',20);
ylabel('sort','fontsize',20);
title(['test accuracy=',num2str(accuracy),'%'],'fontsize',18)
% Error
error_t = mse(yt-Y_wine_test');

set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'wine_rp', 'pdf') %Save figure

figure(3)
plotperform(tr);
set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'wine_rp_per', 'pdf') %Save figure