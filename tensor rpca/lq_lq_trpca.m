function [L,S,out] = lq_lq_trpca(Y,lamda,q1,q2,L0,S0);
% lq_lq_rpca solves
%
%   minimize ||L||_{S_q1} + lamda*||S||_q2 + 1/(2*\mu)*||Y-L-S||_F^2
%
% Inputs 
%	1=>q1,q2=>0
%   L0,S0: intialization
% Outputs
%	L,S: the recovery
%	out.el, out.es: the error with respect to the true


%Convergence setup
MAX_ITER = 1e3;
ABSTOL = 1e-6;

[m, n, c] = size(Y);

%Initialize
if nargin<5
	L = zeros(m,n,c);
    S = zeros(m,n,c);
else
    L = L0;
    S = S0;
end

isquiet = 0;

mu  = norm(Y(:)); 
phi = 0e-5;

out.el = []; out.es = [];

for iter = 1:MAX_ITER

    Lm1 = L; 
    Sm1 = S;
	
    % for acceleration of the algorithm
    mu = mu * 0.97;
    
    % L-update 
    L = prox_lq((Y - S + phi*L)/(1+phi), mu/(1+phi), q1);
       
    % S-update
    T = (Y - L + phi*S)/(1+phi);
    S = reshape(shrinkage_Lq(T(:), q2, mu*lamda/(1+phi), 1), m, n, c);
    
%     [mu/(1+phi), mu*lamda/(1+phi)]

    % debug information
    if ~isquiet
       out.el  = [out.el, norm(L(:)-Lm1(:))/norm(Y(:))];
       out.es  = [out.es, norm(S(:)-Sm1(:))/norm(Y(:))];
    end
        
    % Check for convergence
    if (norm(L(:)-Lm1(:))< sqrt(m*n*c)*ABSTOL) & (norm(S(:)-Sm1(:))< sqrt(m*n*c)*ABSTOL)
        num_stop = num_stop + 1;
        if num_stop==3
            break;
        end
    else
        num_stop = 0;
    end
    
end

end
