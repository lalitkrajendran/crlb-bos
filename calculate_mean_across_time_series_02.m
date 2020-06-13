function [X_mean, Y_mean, Z_mean, d_x_mean, d_y_mean, R_mean, I_mean] = calculate_mean_across_time_series_02(X_all, Y_all, Z_all, d_x_all, d_y_all, R_all, I_all)
    
    X_mean = mean(X_all, 2, 'omitnan');
    Y_mean = mean(Y_all, 2, 'omitnan');
    Z_mean = mean(Z_all, 2, 'omitnan');
    d_x_mean = mean(d_x_all, 2, 'omitnan');
    d_y_mean = mean(d_y_all, 2, 'omitnan');
    R_mean = mean(R_all, 2, 'omitnan');
    I_mean = mean(I_all, 2, 'omitnan');

end