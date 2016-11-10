clc
clear 
close all

%% breast
load 'breast_mod.txt';
X_breast_train=(breast_mod(1:555,1:9));
Y_breast_train=(breast_mod(1:555,10));
X_breast_test=(breast_mod(556:683,1:9));
Y_breast_test=(breast_mod(556:683,10));

X_breast_train=normalization(X_breast_train);
X_breast_test=normalization(X_breast_test);

for i=1:size(Y_breast_train,1)
    if Y_breast_train(i)==2
        Y_breast_train(i)=1;
    else
        Y_breast_train(i)=2;
    end
end

for i=1:size(Y_breast_test,1)
    if Y_breast_test(i)==4
        Y_breast_test(i)=2;
    else
        Y_breast_test(i)=1;
    end
end

for j=1:size(Y_breast_test,1)
    for i=1:size(Y_breast_test,1)-1
        if Y_breast_test(i) > Y_breast_test(i+1)
            temp = Y_breast_test(i);
            tempx = X_breast_test(i,:);
            Y_breast_test(i) = Y_breast_test(i+1);
            X_breast_test(i,:) = X_breast_test(i+1,:);
            Y_breast_test(i+1) = temp;
            X_breast_test(i+1,:) = tempx;
        end
    end
end

net1 = newff(X_breast_train',Y_breast_train',[3 3],{'tansig','logsig','purelin'},'trainscg');
%net1 = newff(X_breast_train',Y_breast_train',[3 3],{'tansig','logsig','purelin'},'trainrp');
net1.trainParam.epochs = 400;
net1.trainParam.show = 25;
net1.trainParam.lr = 0.0015;
net1.trainParam.goal = 0;

[net1,tr] = train(net1, X_breast_train', Y_breast_train');

% Test network
yt = sim(net1,X_breast_test');
plot(yt,'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count=0;
for i=1:length(yt)
    if yt(i)<1.5
    yt(i)=1;
    else
    yt(i)=2;
       end
ERR(i)=Y_breast_test(i)-yt(i);
    if ERR(i)==0
    count=count+1;
    end
end

accuracy=count/length(yt)*100;

figure(2)
plot(1:length(yt),Y_breast_test);
hold on;
plot(1:length(yt),yt,'ro');
xlabel('sample','fontsize',20);
ylabel('sort','fontsize',20);
title(['test accuracy=',num2str(accuracy),'%'],'fontsize',18)
% Error
error_t = mse(yt-Y_breast_test');

set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'breast_scg', 'pdf') %Save figure

figure(3)
plotperform(tr);
set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'breast_scg_per', 'pdf') %Save figure