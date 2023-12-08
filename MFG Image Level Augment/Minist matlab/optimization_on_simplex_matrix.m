function [ x] = optimization_on_simplex_matrix(y,s)
%UNTITLED3 Summary of this function goes here
% % Consider the problem of computing the Euclidean projection of a point y = [y1, . . . , yD]
% ? ? R
% D onto the
% probability simplex, which is defined by the following optimization problem: 
%sun(y) = s;
[m1,m2] = size(y);
size_m  = m1*m2;
z = reshape(y,[size_m,1]);
u=sort(z,'descend');
r=0;
for i =1:size_m
    w = u(i) + 1/i*(s-sum(u(1:i)));
    if w>0
        r=i;
    end
end
lambda = 1/r*(s-sum(u(1:r)));
x = max(y+lambda,0);
end
