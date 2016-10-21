function g = d_sigmoid(x)

g=exp(-x) ./ (( 1.0 + exp(-x) ).^2);

end