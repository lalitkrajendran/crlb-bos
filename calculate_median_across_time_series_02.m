function [X_median, Y_median, Z_median, d_x_median, d_y_median, R_median, I_median] = calculate_median_across_time_series_02(X_all, Y_all, Z_all, d_x_all, d_y_all, R_all, I_all)
    
    X_median = median(X_all, 2, 'omitnan');
    Y_median = median(Y_all, 2, 'omitnan');
    Z_median = median(Z_all, 2, 'omitnan');
    d_x_median = median(d_x_all, 2, 'omitnan');
    d_y_median = median(d_y_all, 2, 'omitnan');
    R_median = median(R_all, 2, 'omitnan');
    I_median = median(I_all, 2, 'omitnan');

end