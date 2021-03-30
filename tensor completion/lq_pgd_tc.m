function [X, out] = lq_pgd_tc(q, Y, P, lamdat, X0, Xtrue);

[m,n,c] = size(Y);

% Lipschitz constant
ka = 1.001;

%Convergence setup
MAX_ITER = 1e3;
ABSTOL = 1e-6;

%Initialize
if nargin<5
	X = zeros(m,n,c);
else
    X = X0;
end

out.e = [];
num_stop = 0;

lamda = 1e3;

for iter = 1 : MAX_ITER

    Xm1 = X;

    if lamda>lamdat % for acceleration
        lamda = lamda*0.97;
    end
    
    X = prox_lq(X - 1/ka*(X - Y).*P, lamda/ka, q);
            
    if nargin>5
%        out.e  = [out.e, norm(X(:)-Xtrue(:),'fro')/norm(Xtrue(:),'fro')];
       out.e  = [out.e, norm(X(:)-Xm1(:),'fro')];
    end
    
    %Check for convergence
    if (norm(X(:)-Xm1(:),'fro')< sqrt(m*n*c)*ABSTOL)
        num_stop = num_stop + 1;
        if num_stop==3
             break;
        end
    else
        num_stop = 0;
    end

end
