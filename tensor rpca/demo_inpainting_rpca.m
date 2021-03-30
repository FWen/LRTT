clear all; clc; close all;

Xgt = double(imread('turtule_img.png'));
[m,n,c] = size(Xgt);

%% synthetic strictly low-rank
for ic=1:c
    [S,V,D] = svd(Xgt(:,:,ic));
    Xgt(:,:,ic) = S(:,1:30)*V(1:30,1:30)*D(:,1:30)';
end

%% random sampling
J = randperm(m*n); 
J = J(1:round(0.2*m*n)); % corruption ratio
for ic=1:c
    X1 = Xgt(:,:,ic); 
    X1(J) = 255;
    X(:,:,ic) = reshape(X1,[m,n]);
end

%% entry-wise noise
SNR = 40;
Y = X + randn(m,n,c) *10^(-SNR/20)*std(X(:));

figure(1); subplot(1,4,1); imshow(uint8(Y));
title('Partial observation');

qs = 0:0.1:1;
lamdas = logspace(-5,1,30);

%% Initialization for nonconvex algorithm
[L0, S0, ~]  = lq_lq_trpca(Y, 1e-2, 1, 1, zeros(size(Y)), zeros(size(Y)));
figure(6); imshow(uint8(L0));

for iq=1:length(qs)
    iq
    parfor k = 1:length(lamdas); 
        k
        [Xr, S, out]  = lq_lq_trpca(Y, lamdas(k), 0.5, qs(iq), L0, S0);
        relerr(iq,k)  = norm(Xr(:)-Xgt(:),'fro')/norm(Xgt(:),'fro');
        PSNRs(iq,k) = psnr(Xr, Xgt, 255);
        X_rs(:,iq,k) = Xr(:);
    end
end

v0 = min(PSNRs(:)); v1 = max(PSNRs(:));
figure(6);
contourf(qs,lamdas,PSNRs',[v0:5:v1]); colorbar; ylabel('\lambda'); xlabel('q');
set(gca, 'CLim', [v0, v1]); set(gca,'yscale','log');

[w1, e1] = max(PSNRs'); [~, lo] = max(w1); ko = e1(lo);
figure(6); hold on;
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
