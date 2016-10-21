function  [cost,w1,w2,w3,b1,b2,b3] = train_part1_hh(weight1,weight2,weight3,bias1,bias2,bias3,X,desired_output)

num_labels = size(weight3, 2);

%% foward
node_a = X * weight1 + bias1;
%Y_node_a = sigmoid(node_a);
Y_node_a = tanh(node_a);
node_b = Y_node_a * weight2 + bias2;
Y_node_b = tanh(node_b);
%Y_node_b = sigmoid(node_b);
node_c = Y_node_b * weight3 + bias3;
Y_output = node_c;

desired_output_mod=zeros(1,num_labels);
desired_output_mod(1,desired_output)=1;

%cost = 0.5 * sum((desired_output_mod - Y_output).^2);
cost = 1;
%% back propagation
mu=0.3;
%% update output neurons
del_w3 = mu * (Y_node_b)' * (desired_output_mod - Y_output);
w3 = weight3 + del_w3;
%% update output bias
del_b3 = mu * (desired_output_mod - Y_output);

b3 = bias3 + del_b3;
%% update weight2 for second hidden neurons
del_w2 = zeros(size(bias1,2),size(bias2,2));
del_b2 = zeros(1,size(bias2,2));
    for j=1:size(bias1,2)
        for k=1:size(bias2,2)
            for m=1:size(bias3,2)
                %del_w2(j,k) = del_w2(j,k) + (desired_output_mod(1,m) - Y_output(1,m)) * weight3(k,m) * d_sigmoid(node_b(1,k)) * (Y_node_a(1,j));
                %del_b2(1,k) = del_b2(1,k) + (Y_output(1,m) - desired_output_mod(1,m)) * weight3(k,m) * d_sigmoid(node_b(1,k));
                del_w2(j,k) = del_w2(j,k) + (desired_output_mod(1,m) - Y_output(1,m)) * weight3(k,m) * d_hyperbolic_tangent(node_b(1,k)) * (Y_node_a(1,j));
                del_b2(1,k) = del_b2(1,k) + (Y_output(1,m) - desired_output_mod(1,m)) * weight3(k,m) * d_hyperbolic_tangent(node_b(1,k));
            end
        end
     end
    
w2 = weight2 + mu * del_w2;
%% update bias2 for second hidden neurons
 
b2 = bias2 - mu *del_b2;
%% update weight1 for first hidden neurons
del_w1 = zeros(size(X,2),size(bias1,2));
del_b1 = zeros(1,size(bias1,2));
    for p=1:size(X,2)
        for j=1:size(bias1,2)
            for k=1:size(bias2,2)
                for m=1:size(bias3,2)
%                     del_w1(p,j) = del_w1(p,j) + ( desired_output_mod(1,m) - Y_output(1,m)) * d_hyperbolic_tangent(node_a(1,j)) ...
%                         * X(1,p) * weight3(k,m) * weight2(j,k) * d_hyperbolic_tangent(node_b(1,k));
%                     del_b1(1,j) = del_b1(1,j) + (desired_output_mod(1,m) - Y_output(1,m)) * d_hyperbolic_tangent(node_a(1,j)) * weight3(k,m) ...
%                      * weight2(j,k) * d_hyperbolic_tangent(node_b(1,k));
                    del_w1(p,j) = del_w1(p,j) + ( desired_output_mod(1,m) - Y_output(1,m)) * d_hyperbolic_tangent(node_a(1,j)) ...
                        * X(1,p) * weight3(k,m) * weight2(j,k) * d_hyperbolic_tangent(node_b(1,k));
                    del_b1(1,j) = del_b1(1,j) + (desired_output_mod(1,m) - Y_output(1,m)) * d_hyperbolic_tangent(node_a(1,j)) * weight3(k,m) ...
                     * weight2(j,k) * d_hyperbolic_tangent(node_b(1,k));
                end
            end
        end
    end

w1 = weight1 + mu *del_w1;
%% update bias1 for first hidden neurons
 
b1 = bias1 + mu * del_b1;
end