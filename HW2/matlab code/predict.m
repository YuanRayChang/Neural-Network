function p = predict(w1,w2,w3,b1,b2,b3,X)
 b1  = repmat(b1,size(X,1),1);
 b2  = repmat(b2,size(X,1),1);
 b3  = repmat(b3,size(X,1),1);
p = zeros(size(X, 1), 1);
node_a = X * w1 + b1;
Y_node_a = tanh(node_a);
%Y_node_a = sigmoid(node_a);
node_b = Y_node_a * w2 + b2;
Y_node_b = tanh(node_b);
%Y_node_b = sigmoid(node_b);
node_c = Y_node_b * w3 + b3;
Y_output = node_c;

[dummy, p] = max(Y_output, [], 2);

end