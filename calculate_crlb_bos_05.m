function [err_var_x, err_var_y, d2_x, d2_y] = calculate_crlb_bos_05(im, n_0, rho_x, rho_y, rho_xx, rho_xy, rho_yy, d_diff, magnification, f_number, Z_D, Z_g, pixel_pitch, noise_gray_level)
% =========================================================================
% This function calculate the Cramer-Rao Lower Bound for a dot viewed
% through a density gradient field using the model provided in Rajendran
% et. al. (2020)
%
% Rajendran et. al. (2020): Uncertainty amplification due to density/refractive-index gradients in Background-Oriented Schlieren experiments
% 
% INPUT:
% im: dot image [gray-level]
% n_0: ambient refractive index
% rho_x, rho_y: density gradient experienced by the angular bisector
%               originating from the dot. [kg/m^4]
% rho_xx, rho_xy, rho_yy: second derivatives of the density field 
%                         experienced by the angular bisector
%                         originating from the dot. [kg/m^5]
% d_dff: diffraction diameter [pix.]
% magnification: magnification of the dot pattern [unitless]
% f_number: fnumber of the setup [unitless]
% Z_D: distance between dot pattern and mid-plane of density field [m]
% Z_g: thickness of the density field [m]
% pixel_pitch: size of a pixel on the camera sensor [m]
% noise_gray_level: standard deviation of the Gaussian noise distribution.
%                   Same units as im [gray-level]
%
% OUTPUT:
% err_var_x, err_var_y: position estimation variance along x and y [pix.^2]
% d2_x, d2_y: square of dot diameters along x and y
%
% AUTHOR:
% Lalit Rajendran (lrajendr@purdue.edu)
% =========================================================================

    % gladstone dale constant
    K = 0.225e-3;   

    % distance of the exit point of the light ray from the dot pattern
    zeta = Z_D; % + Z_g/2;
    
    % calculate first term containing noise level and exposure
%     term_1 = 2 * sqrt(2*pi) * noise_level * max(im(:)) / sum(im(:));
    term_1 = 2 * sqrt(2*pi) * noise_gray_level / sum(im(:));
    
    % standard deviation corresponding to diffraction diameter
    term_2 = d_diff^2/16;
    
    
    % calculate term containing the sensitivity
    term_3_a = (magnification * K/1.000278 * Z_D * Z_g); %zeta^2);
    
    % calculate second derivatives of density in camera co-ordinate system
    rho_xi_xi = rho_xx;
    rho_eta_eta = rho_yy;
    rho_xi_eta = rho_xy;
    
    % calculate term containing density gradients
    term_3_b_x = sqrt(rho_xi_xi^2 + rho_xi_eta^2); 
    term_3_b_y = sqrt(rho_xi_eta^2 + rho_eta_eta^2); 
    
    % calculate mean deflection
    term_3_d =  K/n_0 * mean(abs(rho_x) + abs(rho_y)) * Z_g;

    % calculate term containing viewing angle
    term_3_c = zeta * (1/f_number * 1/(1 + 1/magnification) - term_3_d);
    
    % calculate blurring due to density gradients
    term_3_x = 1/12 * 1/pixel_pitch^2 * term_3_a^2 * term_3_b_x^2 * term_3_c^2;
    term_3_y = 1/12 * 1/pixel_pitch^2 * term_3_a^2 * term_3_b_y^2 * term_3_c^2;
    
    % calculate standard deviations along x and y
    eta_x = sqrt(term_2 + term_3_x);
    eta_y = sqrt(term_2 + term_3_y);

    % calculate total dot diameter in the gradient image
    d2_x = 16 * eta_x^2;
    d2_y = 16 * eta_y^2;
    
    %% calculate covariance term
    
    cov_term = 1/12 * 1/pixel_pitch^2 * term_3_a^2 * rho_xi_eta * (rho_xi_xi + rho_eta_eta) * term_3_c^2;
    
    % calculate correlation coefficient
    R = cov_term/(eta_x * eta_y);

    %% calculate position estimation variance

    err_var_x = (term_1 * (eta_x)^(3/2) * (eta_y)^(1/2) * (1 - R^2)^0.25)^2;
    err_var_y = (term_1 * (eta_x)^(1/2) * (eta_y)^(3/2) * (1 - R^2)^0.25)^2;

end