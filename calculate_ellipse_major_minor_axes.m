function [d1, d2] = calculate_ellipse_major_minor_axes(covariance_matrix)

    % calculate eigen values of the covariance matrix
    [~,S,~]=svd(covariance_matrix);
    Sig123=diag(S)';

    % Principal axis diameters
    d1=4*(Sig123(1)^0.5);
    d2=4*(Sig123(2)^0.5);

end