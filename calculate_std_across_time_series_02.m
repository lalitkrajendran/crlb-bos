function [X_std, Y_std, Z_std, d_x_std, d_y_std, R_std, I_std] = calculate_std_across_time_series_02(X_all, Y_all, Z_all, d_x_all, d_y_all, R_all, I_all)
    
    X_std = std(X_all, [], 2, 'omitnan');
    Y_std = std(Y_all, [], 2, 'omitnan');
    Z_std = std(Z_all, [], 2, 'omitnan');
    d_x_std = std(d_x_all, [], 2, 'omitnan');
    d_y_std = std(d_y_all, [], 2, 'omitnan');
    R_std = std(R_all, [], 2, 'omitnan');
    I_std = std(I_all, [], 2, 'omitnan');

end