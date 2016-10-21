clear ; close all; clc
%% part2
input_layer_size=2;
first_hidden_layer_size=3;
second_hidden_layer_size=2;
output_layer_size=2;

% initial_weight1 = rand(input_layer_size, first_hidden_layer_size);
% initial_weight2 = rand(first_hidden_layer_size, second_hidden_layer_size);
% initial_weight3 = rand(second_hidden_layer_size, output_layer_size);
% initial_bias1   = rand(1,first_hidden_layer_size);
% initial_bias2   = rand(1,second_hidden_layer_size);
% initial_bias3   = rand(1,output_layer_size);

initial_weight1 = [0.449476830054760,0.588608820769942,0.702758034792588;0.355275715183646,0.450249360391824,0.00175025992302724];
initial_weight2 = [0.532256025449206,0.889381696633984;0.540508973155255,0.251460263311904;0.742667597547335,0.767436593297423];
initial_weight3 = [0.127640340002447,0.404717500605161;0.0380866871003673,0.173506839057552];
initial_bias1   = [0.814107705602788,0.490895763512879,0.403942984665026];
initial_bias2   = [0.925142817859544,0.781108452153246];
initial_bias3   = [0.655817200101824,0.895204760628678];

w1 = initial_weight1;
w2 = initial_weight2;
w3 = initial_weight3;
b1 = initial_bias1;
b2 = initial_bias2;
b3 = initial_bias3;

X1 = rand(500,1);
X2 = rand(500,1);
for k = 501:1000
    X1(k,1) = sin((k*pi)/45);
    X2(k,1) = sin((k*pi)/45);
end

X = [X1 X2];
Y1 = X1.^2 + X2.^2;
Y2 = (X1.^2 + X2.^2)/3;
Y = [Y1 Y2];
disp 'start training part2';
iteration=1000;
average_error = zeros(iteration,2);
total_error = zeros(1,2);

for j=1:iteration  %50
for i=1:size(X,1)
[cost,w1,w2,w3,b1,b2,b3] = train_part2(w1,w2,w3,b1,b2,b3,X(i,:),Y(i,:));
total_error = total_error + cost;
end
average_error(j,:) = total_error / size(X,1);
total_error = 0;
fprintf('%f  %',j/iteration); clc;
end
disp 'finish training part2';

X1_test = rand(50,1);
X2_test = rand(50,1);
for k = 51:100
    X1_test(k,1) = 1.05 * sin(pi*k/5);
    X2_test(k,1) = 1.05 * sin(pi*k/5);
end
X_test = [X1_test X2_test];
Y1_test = X1_test.^2 + X2_test.^2;
Y2_test = (X1_test.^2 + X2_test.^2)/3;
Y_test = [Y1_test Y2_test];

b1   = repmat(b1,size(X_test,1),1);
b2   = repmat(b2,size(X_test,1),1);
b3   = repmat(b3,size(X_test,1),1);
node_a = X_test * w1 + b1;
Y_node_a = tanh(node_a);
node_b = Y_node_a * w2 + b2;
Y_node_b = sigmoid(node_b);
node_c = Y_node_b * w3 + b3;
Y_output = node_c;

x_axis = 1:100;
figure;
plot(x_axis,Y_test(:,1),'r',x_axis,Y_output(:,1),'b--');
title('Y1');
legend('desired output','prediction','location','northwest');
set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'y1', 'pdf') %Save figure

figure;
plot(x_axis,Y_test(:,2),'r',x_axis,Y_output(:,2),'b--');
title('Y2');
legend('desired output','prediction','location','northwest');
set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'y2', 'pdf') %Save figure