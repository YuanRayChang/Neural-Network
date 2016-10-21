function  [X_nor]=normalization(X)

for i=1:size(X,2)
sigma=std(X(:,i));
X(:,i)=(X(:,i)-mean(X(:,i)))/sigma;
X_nor=X;
end
end
