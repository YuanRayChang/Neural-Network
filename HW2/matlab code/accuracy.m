function rate = accuracy(X,Y)

index = X == Y;

rate = nnz(index) / size(X,1);

end