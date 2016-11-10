clc
clear 
close all

%% yeast
load 'yeast_mod.txt'

X_yeast_train=[(yeast_mod(1:370,1:8));(yeast_mod(464:806,1:8));(yeast_mod(893:1087,1:8)); ...
    (yeast_mod(1137:1266,1:8));(yeast_mod(1300:1340,1:8));(yeast_mod(1351:1385,1:8)); ...
    (yeast_mod(1395:1422,1:8));(yeast_mod(1430:1453,1:8));(yeast_mod(1460:1475,1:8)); ...
    (yeast_mod(1480:1483,1:8));];
Y_yeast_train=[(yeast_mod(1:370,9));(yeast_mod(464:806,9));(yeast_mod(893:1087,9)); ...
    (yeast_mod(1137:1266,9));(yeast_mod(1300:1340,9));(yeast_mod(1351:1385,9)); ...
    (yeast_mod(1395:1422,9));(yeast_mod(1430:1453,9));(yeast_mod(1460:1475,9)); ...
    (yeast_mod(1480:1483,9));];
X_yeast_test=[(yeast_mod(371:463,1:8));(yeast_mod(807:892,1:8));(yeast_mod(1088:1136,1:8)); ...
    (yeast_mod(1267:1299,1:8));(yeast_mod(1341:1350,1:8));(yeast_mod(1386:1394,1:8)); ...
    (yeast_mod(1423:1429,1:8));(yeast_mod(1454:1459,1:8));(yeast_mod(1476:1479,1:8)); ...
    (yeast_mod(1484,1:8));];
Y_yeast_test=[(yeast_mod(371:463,9));(yeast_mod(807:892,9));(yeast_mod(1088:1136,9)); ...
    (yeast_mod(1267:1299,9));(yeast_mod(1341:1350,9));(yeast_mod(1386:1394,9)); ...
    (yeast_mod(1423:1429,9));(yeast_mod(1454:1459,9));(yeast_mod(1476:1479,9)); ...
    (yeast_mod(1484,9));];
X_yeast_train = [X_yeast_train; ...
    X_yeast_train(1039:1186,:); X_yeast_train(1039:1186,:) ;X_yeast_train(1039:1186,:); ...
    X_yeast_train(1115:1186,:); X_yeast_train(1115:1186,:); X_yeast_train(1115:1186,:)];
Y_yeast_train = [Y_yeast_train; ...
    Y_yeast_train(1039:1186,:); Y_yeast_train(1039:1186,:) ;Y_yeast_train(1039:1186,:); ...
    Y_yeast_train(1115:1186,:); Y_yeast_train(1115:1186,:); Y_yeast_train(1115:1186,:)];

X_yeast_train=normalization(X_yeast_train);
X_yeast_test=normalization(X_yeast_test);

p = randperm(size(X_yeast_train,1));
X_yeast_train=X_yeast_train(p,:);
Y_yeast_train=Y_yeast_train(p,:);

desired_output_mod=zeros(length(Y_yeast_train),10);
for i=1:length(Y_yeast_train)
  desired_output_mod(i,Y_yeast_train(i))=1;
end

%net1 = newff(X_yeast_train',desired_output_mod',[5 7],{'tansig','logsig','purelin'},'trainscg');
net1 = newff(X_yeast_train',desired_output_mod',[5 7],{'tansig','logsig','purelin'},'trainrp');
net1.trainParam.epochs = 400;
net1.trainParam.show = 25;
net1.trainParam.lr = 0.0015;
net1.trainParam.goal = 0;

[net1,tr] = train(net1, X_yeast_train', desired_output_mod');

% Test network
yt = sim(net1,X_yeast_test');

[dummy, p] = max(yt, [], 1);
plot(p,'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count=0;
 for i=1:length(yt)

ERR(i)=Y_yeast_test(i)-p(i);
    if ERR(i)==0
    count=count+1;
    end
 end
 
accuracy=count/length(yt)*100;

figure(2)
plot(1:length(yt),Y_yeast_test);
hold on;
plot(1:length(yt),p,'ro');
xlabel('sample','fontsize',20);
ylabel('sort','fontsize',20);
title(['test accuracy=',num2str(accuracy),'%'],'fontsize',18)
% Error
%error_t = mse(yt-Y_yeast_test');

set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'yeast_rp', 'pdf') %Save figure

figure(3)
plotperform(tr);
set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'yeast_rp_per', 'pdf') %Save figure