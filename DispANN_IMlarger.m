function [D_expected, D_nonzero, sigma_lnD, P_zero] = DispANN_IMlarger(input_PGA,input_SA2s,input_Ky,input_Ts,typeDisp)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com;  wangmx@whu.edu.cn)
% February 2023
%
% Predict the seismically-induced median and maximum sliding displacements
% (i.e., D50 and D100) over all horizontal ground motion orientations using
% PGA and SA(2 s) defined as the larger intensity value of the two as-recorded
% ground motion components.
%
% This code is supported by the following manuscript:
% Wang M.X., Leung Y.F., Wang G., and Zhang P. (2023). "Semi-empirical
% predictive models for seismically-induced slope displacements considering
% ground motion directionality.", which has been submitted to
% ASCE Journal of Geotechnical and Geoenvironmental Engineering.
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
        X_min = [-5.168 -7.608 -4.605 -4.605];
        X_max = [0.904 0.282 -0.223 0.693];
        weightMatrix = [
            4.042 	-1.306 	0.625 	-0.393 	-0.133 	1.030 	1.722 	0.631 	-0.913 	2.506 	3.568 	0.868 	0.998 	3.428 	3.110 	0.309 	-0.749 	0.063 	2.922 	1.279 	-0.700 	0.085
            -0.453 	5.275 	-2.467 	1.360 	8.074 	2.217 	-0.839 	-0.183 	0.344 	1.214 	1.351 	-0.537 	-0.802 	-0.456 	3.211 	2.457 	4.977 	1.621 	0.474 	3.485 	6.604 	2.777
            -2.071 	-1.466 	-0.921 	-0.061 	0.496 	-1.834 	-0.124 	-0.196 	-1.637 	-2.609 	-2.743 	1.111 	0.442 	-1.647 	-0.863 	-0.443 	0.571 	-1.237 	-2.677 	-4.052 	-3.649 	0.095
            2.813 	0.817 	-0.727 	-0.847 	-0.021 	0.015 	0.867 	1.106 	-2.973 	-0.632 	0.621 	2.445 	1.937 	2.515 	-0.022 	-0.072 	4.616 	-0.191 	-0.083 	-4.638 	-3.288 	-0.380
            ];
        weightVector = [
            -1.626 	0.482 	4.194 	2.060 	-0.447 	1.539 	2.836 	3.502 	2.405 	-3.146 	1.105 	7.232 	-9.609 	2.154 	0.165 	1.723 	0.481 	-1.928 	2.536 	0.619 	0.278 	-0.951
            ];
        biasVector = [
            -1.204 	-3.643 	2.952 	-0.885 	-4.543 	-1.714 	-1.967 	-1.246 	1.080 	0.081 	-0.869 	-1.029 	-1.264 	-0.841 	-1.102 	-1.656 	-6.441 	-0.882 	-0.016 	1.031 	-0.788 	-0.644
            ];
        biasScalar = 1.206;
        % polynomial coefficients for standard deviation
        c = [0.318	1.208	-1.109	0.292];
        % polynomial coefficients for zero-displacement probability
        a = [3.626 	-5.609 	-0.798 	7.380 	-13.497 	0.344
            4.196 	-1.577 	-2.832 	8.473 	5.275 	0.445];
    case 'D100'
        % ANN coefficients for nonzero displacement
        X_min = [-5.232 -7.608 -4.605 -4.605];
        X_max = [0.904 0.282 -0.223 0.693];
        weightMatrix = [
            0.645 	2.082 	-1.810 	2.662 	3.638 	0.733 	5.541 	1.228 	-0.087 	2.690 	0.081 	4.229 	1.381 	0.391 	2.669 	-0.673 	-1.897 	2.441 	1.394 	-0.020 	0.495 	-1.071
            -0.849 	0.785 	5.015 	-1.598 	1.581 	1.531 	1.199 	0.377 	4.917 	1.996 	2.253 	-0.365 	2.529 	3.465 	1.057 	1.080 	3.603 	-2.471 	-0.048 	2.154 	-0.840 	5.134
            1.469 	-2.391 	-0.511 	-0.373 	-3.003 	-1.077 	-3.520 	-2.104 	1.886 	-2.645 	-1.014 	-2.756 	-3.819 	-2.190 	-1.442 	0.220 	-0.783 	0.663 	-0.427 	-0.217 	1.849 	-1.479
            2.929 	-1.289 	3.817 	3.056 	0.160 	-0.887 	1.715 	-1.074 	3.761 	-0.031 	-1.304 	0.059 	-4.254 	-1.336 	0.575 	-0.801 	-0.843 	3.877 	-0.291 	0.192 	2.988 	0.563
            ];
        weightVector = [
            -6.575 	-4.219 	0.445 	0.473 	2.175 	1.794 	0.446 	2.671 	-3.479 	-2.586 	-1.584 	1.752 	0.972 	0.792 	0.584 	1.627 	0.491 	2.600 	1.898 	0.647 	4.699 	0.556
            ];
        biasVector = [
            -1.967 	0.865 	-5.650 	0.138 	-0.860 	-0.956 	-0.593 	0.861 	-8.869 	-0.670 	-0.625 	0.790 	1.421 	-0.600 	-1.227 	1.008 	-2.970 	-4.501 	-2.048 	-0.524 	-1.734 	-3.186
            ];
        biasScalar = 0.512;
        % polynomial coefficients for standard deviation
        c = [0.318	1.289	-1.214	0.325];
        % polynomial coefficients for zero-displacement probability
        a = [2.111 	-5.676 	-0.748 	7.141 	-13.272 	0.323
            3.386 	-1.607 	-2.730 	8.508 	5.262 	0.469];
end
boundlnD_sigma = [-0.5, 6];  % boundary lnD values for estimating standard deviation

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
sigma_lnD(log(D_nonzero)<boundlnD_sigma(1)) = c(1)+c(2)./boundlnD_sigma(1)+c(3)./boundlnD_sigma(1).^2+c(4)./boundlnD_sigma(1).^3;
sigma_lnD(log(D_nonzero)>boundlnD_sigma(2)) = c(1)+c(2)./boundlnD_sigma(2)+c(3)./boundlnD_sigma(2).^2+c(4)./boundlnD_sigma(2).^3;

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
