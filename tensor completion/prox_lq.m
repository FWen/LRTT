function X = prox_lq(Y, lamda, q);

% Solve the tensor Schatten-q norm regularized proximal minimization 
% min_X lam*||X||_{S_q}+1/2*||X-Y||_F^2


[n1,n2,n3] = size(Y);
Y = fft(Y,[],3);
X = zeros(n1,n2,n3);
        
% the first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = shrinkage_Lq(diag(S), q, lamda, 1);
indx = find(S>0);
X(:,:,1) = U(:,indx)*diag(S(indx))*V(:,indx)';

for k = 2:round(n3/2)
    [U,S,V] = svd(Y(:,:,k),'econ');
    S = shrinkage_Lq(diag(S), q, lamda, 1);
    indx = find(S>0);
    X(:,:,k) = U(:,indx)*diag(S(indx))*V(:,indx)';
    X(:,:,n3+2-k) = conj(X(:,:,k));
end

if mod(n3,2) == 0
    [U,S,V] = svd(Y(:,:,k+1),'econ');
    S = shrinkage_Lq(diag(S), q, lamda, 1);
    indx = find(S>0);
    X(:,:,k+1) = U(:,indx)*diag(S(indx))*V(:,indx)';
end
X = ifft(X,[],3);
