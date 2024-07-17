function [D_expected, D_nonzero, sigma_lnD, P_zero] = DispANN_IM50(input_PGA,input_SA2s,input_Ky,input_Ts,typeDisp)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com;  wangmx@whu.edu.cn)
% February 2023
%
% Predict the seismically-induced median and maximum sliding displacements
% (i.e., D50 and D100) over all horizontal ground motion orientations using
% PGA and SA(2 s) defined as RotD50 (median value over all orientations).
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   input_PGA      = Vector or matrix of inputted PGA (in units of g)
%   input_SA2s     = Vector or matrix of inputted SA(2 s) (in units of g)
%   input_Ky       = Vector or matrix of inputted yield acceleration (in units of g)
%   input_Ts       = Vector or matrix of inputted fundamental period (in units of s)
%   (these inputs must be in the same dimension)
%   typeDisp       = 'D50' representing the 50th percentile displacement over all ground motion orientations
%                  = 'D100' representing the 100th percentile displacement over all ground motion orientations
%
% OUTPUT
%
%   D_expected     = Expectation of displacement (in units of cm) considering the effects of small displacement 
%   D_nonzero      = Nonzero displacement (in units of cm)
%   sigma_lnD      = Standard deviation of logarithmic displacement
%   P_zero         = Probability of the occurrence of 'zero' displacement
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% specify coefficients for corresponding models
switch typeDisp
    case 'D50'
        % ANN coefficients for nonzero displacement
        X_min = [-5.294 -7.651 -4.605 -4.605];
        X_max = [0.570 0.116 -0.223 0.693];
        weightMatrix = [
            0.504 	2.854 	0.578 	-0.091 	1.674 	2.300 	0.100 	0.310 	0.101 	0.420 	-1.559 	1.517 	2.372 	-0.156 	0.151 	0.936 	0.499 	0.666 	4.821 	2.743 	2.107 	-1.071
            -3.232 	0.383 	-1.661 	-1.513 	-0.790 	0.123 	-1.259 	1.700 	3.762 	-0.282 	5.546 	3.335 	-0.765 	3.338 	3.953 	2.338 	-0.247 	2.983 	2.022 	0.797 	0.867 	2.517
            2.498 	-2.007 	0.691 	3.151 	-0.321 	-1.704 	-3.569 	-1.375 	-1.498 	0.417 	-0.966 	-2.745 	-0.848 	-1.488 	-1.601 	-2.500 	-1.062 	-0.415 	-4.023 	-2.749 	-1.929 	1.671
            5.058 	0.869 	2.472 	7.357 	1.357 	0.336 	-5.399 	-0.567 	0.749 	0.112 	3.614 	0.134 	1.240 	-0.549 	-0.593 	-2.727 	-0.025 	0.614 	0.074 	-0.792 	-0.538 	7.506
            ];
        weightVector = [
            1.995 	2.944 	0.878 	-1.153 	-5.819 	-5.051 	1.699 	1.739 	0.973 	3.833 	0.435 	0.779 	3.360 	-2.565 	1.442 	-0.584 	2.922 	0.479 	0.478 	1.577 	1.947 	0.666
            ];
        biasVector = [
            -3.825 	-0.524 	0.649 	-4.692 	-1.939 	-0.637 	6.169 	-2.118 	-3.021 	-0.456 	-5.741 	-2.534 	-2.979 	-2.110 	-2.596 	-0.660 	0.561 	-0.920 	-1.917 	-0.608 	-1.433 	-7.388
            ];
        biasScalar = 1.497;
        % polynomial coefficients for standard deviation
        c = [0.294	1.149	-1.046	0.274];
        % polynomial coefficients for zero-displacement probability
        a = [3.078	-8.223	-0.750	9.088	-28.082	0.305
            1.877	-1.773	-2.530	7.302	5.227	0.314];
    case 'D100'
        % ANN coefficients for nonzero displacement
        X_min = [-5.210 -7.651 -4.605 -4.605];
        X_max = [0.570 0.116 -0.223 0.693];
        weightMatrix = [
            -1.774 	-0.631 	0.919 	1.418 	0.100 	2.652 	1.092 	1.347 	-0.015 	4.020 	2.570 	2.805 	-1.328 	-0.674 	0.380 	0.438 	-0.501 	1.539 	2.061 	2.342 	-0.542 	0.422
            5.484 	1.892 	1.337 	-0.357 	-1.187 	-3.694 	0.055 	0.917 	-1.875 	0.325 	0.056 	1.074 	4.841 	-0.543 	1.841 	2.553 	4.918 	2.945 	0.540 	0.360 	4.426 	1.830
            -0.170 	-0.654 	-2.187 	-0.131 	-1.848 	-0.355 	0.319 	-1.974 	2.535 	-2.282 	-0.695 	-2.232 	-1.570 	-0.716 	-3.478 	-1.223 	-2.577 	-2.480 	-1.288 	-1.624 	-2.264 	-3.991
            4.416 	-2.361 	-2.220 	1.264 	-2.204 	2.360 	1.254 	-1.544 	4.438 	2.104 	0.844 	0.135 	0.999 	-1.157 	-6.291 	0.077 	-1.796 	0.064 	-0.039 	0.410 	-1.466 	-6.743
            ];
        weightVector = [
            0.515 	-0.957 	-1.944 	-9.831 	4.539 	1.250 	7.296 	2.651 	3.382 	0.661 	2.952 	2.671 	0.547 	3.373 	3.002 	1.390 	1.856 	0.831 	2.794 	-4.141 	-2.556 	-1.861
            ];
        biasVector = [
            -6.332 	-0.476 	-0.038 	-1.560 	3.699 	-4.649 	-1.101 	0.009 	-3.441 	-0.465 	-3.048 	-0.906 	-3.459 	1.025 	3.501 	-2.063 	-2.956 	-2.189 	-1.248 	-0.924 	-2.551 	3.495
            ];
        biasScalar = -0.114;
        % polynomial coefficients for standard deviation
        c = [0.294	1.245	-1.136	0.297];
        % polynomial coefficients for zero-displacement probability
        a = [1.027	-8.004	-0.666	8.458	-26.633	0.266
            1.122	-1.776	-2.402	7.226	5.166	0.325];
end
boundlnD_sigma = [0.5, 6];  % boundary lnD values for estimating standard deviation 

%% estimate target displacement indexes
% ensure predictors are in vector form
[n_row,n_col] = size(input_PGA);
n_data = n_row*n_col;
x_PGA = reshape(input_PGA,n_data,1);
x_SA2s = reshape(input_SA2s,n_data,1);
x_Ky = reshape(input_Ky,n_data,1);
x_Ts = reshape(input_Ts,n_data,1);

% normalize predictors (equation 2)
X_norm = (1+1)*(log([x_PGA,x_SA2s,x_Ky,x_Ts])-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]))-1;

% make prediction of nonzero displacement (equations 3 and 4)
D_nonzero = exp((2./(1+exp(-2*(X_norm*weightMatrix+repmat(biasVector,[n_data,1]))))-1)*weightVector'+biasScalar);

% calculate standard deviation of lnD (equation 8)
sigma_lnD = c(1)+c(2)./log(D_nonzero)+c(3)./(log(D_nonzero)).^2+c(4)./(log(D_nonzero)).^3;
sigma_lnD(log(D_nonzero)<boundlnD_sigma(1)) = c(1)+c(2)./boundlnD_sigma(1)+c(3)./(boundlnD_sigma(1)).^2+c(4)./(boundlnD_sigma(1)).^3;
sigma_lnD(log(D_nonzero)>boundlnD_sigma(2)) = c(1)+c(2)./boundlnD_sigma(2)+c(3)./(boundlnD_sigma(2)).^2+c(4)./(boundlnD_sigma(2)).^3;

% estimate zero-displacement probability (equation 7)
P_zero = zeros(size(D_nonzero));
indTs1 = x_Ts<=0.2;
indTs2 = x_Ts>0.2;
P_zero(indTs1) = 1./(1+exp(-(a(1,1)+a(1,2)*log(x_PGA(indTs1))+a(1,3)*log(x_SA2s(indTs1))+a(1,4)*log(x_Ky(indTs1))+a(1,5)*x_Ts(indTs1)+a(1,6)*log(x_Ky(indTs1)).^2)));
P_zero(indTs2) = 1./(1+exp(-(a(2,1)+a(2,2)*log(x_PGA(indTs2))+a(2,3)*log(x_SA2s(indTs2))+a(2,4)*log(x_Ky(indTs2))+a(2,5)*x_Ts(indTs2)+a(2,6)*log(x_Ky(indTs2)).^2)));

% estimate expected displacement considering small displacement(equation 6)
D_expected = D_nonzero.*(1-P_zero);

% make dimension of outputs consistent with that of inputs
D_nonzero = reshape(D_nonzero,n_row,n_col);
sigma_lnD = reshape(sigma_lnD,n_row,n_col);
P_zero = reshape(P_zero,n_row,n_col);
D_expected = reshape(D_expected,n_row,n_col);

