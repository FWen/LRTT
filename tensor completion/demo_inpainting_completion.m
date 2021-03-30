clear all; clc; close all;

X = double(imread('turtule_img.png'));
[m,n,c] = size(X);

%% synthetic strictly low-rank
for ic=1:c
    [S,V,D] = svd(X(:,:,ic));
    X(:,:,ic) = S(:,1:30)*V(1:30,1:30)*D(:,1:30)';
end

%% random sampling
J = randperm(m*n); 
J = J(1:round(0.3*m*n)); % Sampling ratio
P = zeros(m*n,1);
P(J) = 1;
P = reshape(P,[m,n]);    % Projection matrix

%% entry-wise noise
SNR = 40;
Y = X.*repmat(P,[1,1,3]) + randn(m,n,c) *10^(-SNR/20)*std(X(:));

figure(1); subplot(1,4,1); imshow(uint8(Y));
title('Partial observation');

qs = 0:0.1:1;
lamdas = logspace(0,5,20);

%% Initialization for nonconvex algorithm
[X0, out]  = lq_pgd_tc(1, Y, P, 1e2, zeros(size(Y)), X);

for iq=1:length(qs)
    iq
    parfor k = 1:length(lamdas); 
        [Xr, out]  = lq_pgd_tc(qs(iq), Y, P, lamdas(k), X0, X);
        relerr(iq,k)  = norm(Xr(:)-X(:),'fro')/norm(X(:),'fro');
        PSNRs(iq,k) = psnr(Xr, X, 255);
        X_rs(:,iq,k) = Xr(:);
    end
end

v0 = min(PSNRs(:)); v1 = max(PSNRs(:));
figure(16);
contourf(qs,lamdas,PSNRs',[v0:0.5:v1]); colorbar; ylabel('\lambda'); xlabel('q');
set(gca, 'CLim', [v0, v1]); set(gca,'yscale','log');

[w1, e1] = max(PSNRs'); [~, lo] = max(w1); ko = e1(lo);
figure(16); hold on;
plot(qs(lo),lamdas(ko),'r*'); hold off;

% Lq with best value of q
X_best = reshape(X_rs(:,lo,ko),[m,n,c]);
figure(1); subplot(1,4,4); imshow(uint8(X_best));
title(['Lq (best q=', num2str(qs(lo), '%.1f'), ', ', 'RelErr=', num2str(relerr(lo,ko), '%.5f'),', ', 'PSNR=', num2str(PSNRs(lo,ko), '%.2f') ' dB)']);

% L0
[RelErr1, mi] = min(relerr(1,:));
X_L0 = reshape(X_rs(:,1,mi),[m,n,c]); 
figure(1); subplot(1,4,2); imshow(uint8(X_L0));
title(sprintf('Hard (RelErr=%.5f, PSNR=%.2f dB)', RelErr1, PSNRs(1,mi)));

% L1 
[RelErr1, mi] = min(relerr(11,:));
X_L1 = reshape(X_rs(:,iq,mi),[m,n,c]); 
figure(1); subplot(1,4,3); imshow(uint8(X_L1));
title(sprintf('Soft (RelErr=%.5f, PSNR=%.2f dB)', RelErr1, PSNRs(iq,mi)));
