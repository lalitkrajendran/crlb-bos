function [X_per, Y_per, Z_per, d_x_per, d_y_per, R_per, I_per] = calculate_per_across_time_series_02(X_all, Y_all, Z_all, d_x_all, d_y_all, R_all, I_all)

    X_per = calculate_percentile_difference(X_all, 2);
    Y_per = calculate_percentile_difference(Y_all, 2);
    Z_per = calculate_percentile_difference(Z_all, 2);
    d_x_per = calculate_percentile_difference(d_x_all, 2);
    d_y_per = calculate_percentile_difference(d_y_all, 2);
    R_per = calculate_percentile_difference(R_all, 2);
    I_per = calculate_percentile_difference(I_all, 2);

end

function y = calculate_percentile_difference(x, dim)
    y = prctile(x, 84, dim) - prctile(x, 16, dim);
end