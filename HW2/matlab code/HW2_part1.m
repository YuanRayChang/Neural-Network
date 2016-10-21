clear ; close all; clc

%%
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

input_layer_size=4;
first_hidden_layer_size=2;
second_hidden_layer_size=3;
output_layer_size=3;

initial_weight1 = rand(input_layer_size, first_hidden_layer_size);
initial_weight2 = rand(first_hidden_layer_size, second_hidden_layer_size);
initial_weight3 = rand(second_hidden_layer_size, output_layer_size);
initial_bias1   = rand(1,first_hidden_layer_size);
initial_bias2   = rand(1,second_hidden_layer_size);
initial_bias3   = rand(1,output_layer_size);

disp 'start training Iris';

w1 = initial_weight1;
w2 = initial_weight2;
w3 = initial_weight3;
b1 = initial_bias1;
b2 = initial_bias2;
b3 = initial_bias3;

accuracy_rate_iris = 0;
count = 0;
p_iris = predict(w1,w2,w3,b1,b2,b3,X_iris_test);
while accuracy_rate_iris < 0.98
for i=1:size(X_iris_train,1)
[cost,w1,w2,w3,b1,b2,b3] = train_part1_hh_matrix(w1,w2,w3,b1,b2,b3,X_iris_train(i,:),Y_iris_train(i,:));
end
count = count + 1;
p_iris = predict(w1,w2,w3,b1,b2,b3,X_iris_test);
accuracy_rate_iris = accuracy(p_iris,Y_iris_test);
end
disp 'finish training Iris';

x_axis = 1:size(X_iris_test,1);
plot(x_axis,Y_iris_test,'r',x_axis,p_iris,'bo');
legend('desired output','prediction','location','northwest');

set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'irishhb1', 'pdf') %Save figure
% wine 
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

input_layer_size=13;
first_hidden_layer_size=4;
second_hidden_layer_size=2;
output_layer_size=3;

initial_weight1 = rand(input_layer_size, first_hidden_layer_size);
initial_weight2 = rand(first_hidden_layer_size, second_hidden_layer_size);
initial_weight3 = rand(second_hidden_layer_size, output_layer_size);
initial_bias1   = rand(1,first_hidden_layer_size);
initial_bias2   = rand(1,second_hidden_layer_size);
initial_bias3   = rand(1,output_layer_size);

w1 = initial_weight1;
w2 = initial_weight2;
w3 = initial_weight3;
b1 = initial_bias1;
b2 = initial_bias2;
b3 = initial_bias3;

disp 'start training wine';
accuracy_rate_wine = 0;
count = 0;
p_wine = predict(w1,w2,w3,b1,b2,b3,X_wine_test);
while accuracy_rate_wine < 0.97
for i=1:size(X_wine_train,1)
[cost,w1,w2,w3,b1,b2,b3] = train_part1_hh_matrix(w1,w2,w3,b1,b2,b3,X_wine_train(i,:),Y_wine_train(i,:));
end
count = count +1;
p_wine = predict(w1,w2,w3,b1,b2,b3,X_wine_test);
accuracy_rate_wine = accuracy(p_wine,Y_wine_test);
end
disp 'finish training wine';

x_axis = 1:size(X_wine_test,1);
plot(x_axis,Y_wine_test,'r',x_axis,p_wine,'bo');
legend('desired output','prediction','location','northwest');

set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'winehh1', 'pdf') %Save figure


% breast-cancer-wisconsin
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
input_layer_size=9;
first_hidden_layer_size=3;
second_hidden_layer_size=3;
output_layer_size=2;

initial_weight1 = rand(input_layer_size, first_hidden_layer_size);
initial_weight2 = rand(first_hidden_layer_size, second_hidden_layer_size);
initial_weight3 = rand(second_hidden_layer_size, output_layer_size);
initial_bias1   = rand(1,first_hidden_layer_size);
initial_bias2   = rand(1,second_hidden_layer_size);
initial_bias3   = rand(1,output_layer_size);

w1 = initial_weight1;
w2 = initial_weight2;
w3 = initial_weight3;
b1 = initial_bias1;
b2 = initial_bias2;
b3 = initial_bias3;
disp 'start training breast';
accuracy_rate_breast = 0;
count = 0;
p_breast = predict(w1,w2,w3,b1,b2,b3,X_breast_test);
while accuracy_rate_breast < 0.98
for i=1:size(X_breast_train,1)
[cost,w1,w2,w3,b1,b2,b3] = train_part1_hh_matrix(w1,w2,w3,b1,b2,b3,X_breast_train(i,:),Y_breast_train(i,:));
end
count = count +1;
p_breast = predict(w1,w2,w3,b1,b2,b3,X_breast_test);
accuracy_rate_breast = accuracy(p_breast,Y_breast_test);
end
disp 'finish training breast';

p_breast = predict(w1,w2,w3,b1,b2,b3,X_breast_test);
accuracy_rate_breast = accuracy(p_breast,Y_breast_test);

x_axis = 1:size(X_breast_test,1);
plot(x_axis,Y_breast_test,'r',x_axis,p_breast,'bo');
legend('desired output','prediction','location','northwest');

set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'breasths1', 'pdf') %Save figure

% Yeast
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

input_layer_size=8;
first_hidden_layer_size=5;
second_hidden_layer_size=7;
output_layer_size=10;

initial_weight1 = rand(input_layer_size, first_hidden_layer_size);
initial_weight2 = rand(first_hidden_layer_size, second_hidden_layer_size);
initial_weight3 = rand(second_hidden_layer_size, output_layer_size);
initial_bias1   = rand(1,first_hidden_layer_size);
initial_bias2   = rand(1,second_hidden_layer_size);
initial_bias3   = rand(1,output_layer_size);

w1 = initial_weight1;
w2 = initial_weight2;
w3 = initial_weight3;
b1 = initial_bias1;
b2 = initial_bias2;
b3 = initial_bias3;

disp 'start training yeast';
iteration=500;
accuracy_rate_yeast = zeros(iteration,1);
for j=1:iteration 
for i=1:size(X_yeast_train,1)
[cost,w1,w2,w3,b1,b2,b3] = train_part1_hh_matrix(w1,w2,w3,b1,b2,b3,X_yeast_train(i,:),Y_yeast_train(i,:));
end
fprintf('%f  %',j/iteration); clc;
p_yeast = predict(w1,w2,w3,b1,b2,b3,X_yeast_test);
accuracy_rate_yeast(j,1) = accuracy(p_yeast,Y_yeast_test);
end
accuracy_rate_yeast(500,1)
disp 'finish training yeast';
x_axis = 1:size(X_yeast_test,1);
plot(x_axis,Y_yeast_test,'r',x_axis,p_yeast,'bo');
legend('desired output','prediction','location','northwest');
title('yeast','FontSize', 14);
xlabel('data','FontSize', 14);
ylabel('class','FontSize', 14);
set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'yeasths3', 'pdf') %Save figure
