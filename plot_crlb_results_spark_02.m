clear
close all
clc

rmpath('/scratch/shannon/a/lrajendr/Software/prana');
restoredefaultpath;

addpath(genpath('/scratch/shannon/c/aether/Projects/BOS/general-codes/matlab-codes/'));
addpath ../dot-tracking-package/
% addpath 
dbstop if error
logical_string = {'False'; 'True'};

%% experiment settings

% date of the test
test_date = '2018-11-19';

% --------------------------
% pulse settings
% --------------------------
% pulse width (ns)
pulse_width = 90;
% pulse voltage (V)
pulse_voltage = 370;
% resistance (ohm)
resistance = 0;
% pulse parameter name
pulse_parameter_name = [num2str(pulse_width) '_' num2str(pulse_voltage) '_R2x' num2str(resistance)]; %'90_370_R2x0';

% --------------------------
% dot pattern settings
% --------------------------
% dot size (mm)
dot_size = 0.042;
% create a string with the dot size to access folders
dot_size_string = ['0_' num2str(dot_size*1e2, '%d') 'mm'];

% --------------------------
% imaging settings
% --------------------------
% magnification (um/pix.)
magnification = 10.5; %10^4/180; %33;
% number of pixels on the camera sensor
y_pixel_number = 704; %1052;
x_pixel_number = 1024;
% size of a pixel on the camera sensor (um)
pixel_pitch = 13.5;
% f-number of the camera aperture

%% processing settings
% ------------------------
% read/write settings
% ------------------------
% top level directory containing images
top_image_directory = fullfile('/scratch/shannon/c/aether/Projects/plasma-induced-flow/analysis/data/spark/', test_date); 
% top level directory to store the results
% top_results_directory = fullfile('/scratch/shannon/c/aether/Projects/plasma-induced-flow/analysis/results/spark/', test_date);
top_results_directory = fullfile('/scratch/shannon/c/aether/Projects/BOS/crlb/analysis/results/spark/', test_date);
% gradient images to be read
image_read_list = [2, 10];
% number of reference images to read
num_ref_images = 100;

% ----------------------------
% general processing settings
% ----------------------------
% perform background subtraction? (true/false)
background_subtraction = false;
% directory containing background image
background_image_directory = '';
% name of the background image file
background_image_filename = '';

% perform minimum subtraction? (true/false)
minimum_subtraction = true;
% gray scale intensity level to subtracted (300 for 0.15 mm, 200 for 0.25
% mm)
minimum_subtraction_intensity_level = 1e4;

% peform image masking? (true/false)
image_masking = true;
% directory containing image mask
image_mask_directory = top_image_directory;
% name of the image mask file
image_mask_filename = 'mask-crlb.tif';

% perform median filtering of dot positions across time series before
% calculating position estimation variance? (true/false)
median_filtering = false;

% ------------------------
% calibration settings
% ------------------------
% directory containing calibration images
calibration_directory = fullfile(top_image_directory, pulse_parameter_name, 'calibration-crlb');
% camera model
camera_model = 'soloff';
% order of the z term in the polynomial mapping function (x and y are
% cubic).
order_z = 1;
% offset in the co-ordinate system of the sensor for synthetic images
% (pix.)
starting_index_x = 0.5; %0;
starting_index_y = 1; %1;

% ------------------------
% identification settings
% ------------------------
% expected dot diameter in the image plane (pix.)
dot_diameter = 5;
% minimum expected area for a group of pixels to be identified as a dot
% (pix.^2)
min_area = 16; %dot_diameter^2 * 0.5;
% expected dot spacing in the image plane (pix.)
dot_spacing = 5;
% subpixel fit to used for centroid estimation
subpixel_fit = 'lsg'; %'tpg';
% use intensity weighted centroid estimates if gaussian fits fail?
% (true/false)
default_iwc = true;
% weights for area, intensity and distance to be used for multi-parametric
% identification
W_area = 1;
W_intensity = 1;
W_distance = 1;

% -------------------
% tracking settings
% -------------------
% intensity, distance and diameter weights for multi-parametric tracking
weights = [1, 1, 1];
% search radius for nearest neighbor tracking (pix.)
s_radius = 2;

%% plot settings

% save figures? (true/false)
save_figures = true;
% minimum allowable position uncertainty [pix.]
min_uncertainty_threshold = 1e-3;
% maximum allowable position uncertainty [pix.]
max_uncertainty_threshold = 0.2;
% bins for position uncertainty histogram
edges = linspace(0, max_uncertainty_threshold, 100);

%% load image filenames

% load list of runs for the current pressure condition
[runs, num_runs] = get_directory_listing(fullfile(top_image_directory, pulse_parameter_name), 'test*');
% run index
test_index = 2;

% directory to store results for the current case
current_results_directory = fullfile(top_results_directory, pulse_parameter_name, runs(test_index).name, ['bg_subtraction=' logical_string{background_subtraction + 1} '_median_filtering=' logical_string{median_filtering+1} '_min_subtraction=' logical_string{minimum_subtraction + 1} '_subpixel_fit=' subpixel_fit '_default_iwc=' logical_string{default_iwc + 1} '_fixed_ref_locxy_ell']);
% directory to store figures
figure_save_directory = fullfile(current_results_directory, 'figures');
mkdir_c(figure_save_directory);

%%
% loop through all images
for image_index = [2, 10] %image_read_list
    %% load workspace to file
    fprintf('loading workspace\n');
    workspace_save_directory = fullfile(current_results_directory, ['im' num2str(image_index, '%04d')], 'workspace');
    
    filename = 'batch_calculate_crlb_spark_03.mat';
    
    load(fullfile(workspace_save_directory, filename));

    %% create directory to store figures for the current case
    if save_figures
        figure_save_directory = fullfile(current_results_directory, ['im' num2str(image_index, '%04d')], 'figures');
        mkdir_c(figure_save_directory);
    end
    
    %% plot interpolated displacements
    
    [X_grid, Y_grid, U_grid, V_grid] = interpolate_tracks(X_ref_tracked, Y_ref_tracked, U, V, dot_spacing);
    
    % display contours
    cmin_current = 0;
    cmax_current = 1;
    contour_levels = linspace(cmin_current, cmax_current, 100);
    figure
    contourf(X_grid, Y_grid, sqrt(U_grid.^2 + V_grid.^2), contour_levels, 'edgecolor', 'none')
    colormap(flipud(gray));
    h2 = colorbar;
    caxis([cmin_current, cmax_current]);
    annotate_image(gcf, gca);
    title(h2, '(pix.)')
    title('Displacement');
    set(gcf, 'Position', [360   584   441   316])
    
    if save_figures
        save_figure_to_png_eps_fig(figure_save_directory, 'displacement', [1, 0, 0]);
    end
    
    
    %% calculate displacement gradients and plot them
    
    [X_grid, Y_grid, dU_dx, dU_dy, dV_dx, dV_dy] = calculate_displacement_gradients_scattered(X_ref_tracked, Y_ref_tracked, U, V, dot_spacing);
    
    % set contour levels
    cmin_current = 0;
    cmax_current = 0.1;
    contour_levels = linspace(cmin_current, cmax_current, 100);
    
    % display results
    figure
    contourf(X_grid, Y_grid, abs(dU_dx + dV_dy), contour_levels, 'edgecolor', 'none')
    colormap(flipud(gray));
    h2 = colorbar;
    caxis([cmin_current, cmax_current]);
    annotate_image(gcf, gca);
    set(gcf, 'Position', [360   584   441   316])
    title('|dU/dx + dV/dy|, (pix./pix.)');
    
    % save figure
    if save_figures
        save_figure_to_png_eps_fig(figure_save_directory, 'laplacian', [1, 0, 0]);
    end
    
    %% plot spatial variaion of the amplification ratio magnitude
   
    % interpolate results onto a grid
    [X_ref_grid, Y_ref_grid, AR_x_grid, AR_y_grid] = interpolate_tracks(X_ref_tracked, Y_ref_tracked, AR_x_tracked, AR_y_tracked, dot_spacing);

    % display contours
    cmin_current = 1.5;
    cmax_current = 1.9;
    contour_levels = linspace(cmin_current, cmax_current, 100);
    figure
    contourf(X_ref_grid, Y_ref_grid, sqrt(AR_x_grid.^2 + AR_y_grid.^2), contour_levels, 'edgecolor', 'none')
    colorcet('fire', 'N', 100, 'reverse', 1);
    caxis([cmin_current, cmax_current]);
    colorbar
    annotate_image(gcf, gca);
    axis([xmin_mask xmax_mask ymin_mask ymax_mask])
    box on
    title('Amplification Ratio')
    set(gcf, 'Position', [360   584   441   316])

    if save_figures
        save_figure_to_png_eps_fig(figure_save_directory, 'amplification_ratio', [1, 0, 0]);
    end
    
    %% plot pdf of position uncertainty

    x_max = 0.04;
    y_max = 150;

    figure
    % plot reference uncertainty
    l1 = area(edges(1:end-1), N_ref);
    l1.FaceColor = 'b';
    l1.FaceAlpha = 0.5;
    l1.LineWidth = 2.0;    
    hold on    
    % plot gradient uncertainty
    l2 = plot(rms_ref_all * [1, 1], [0, y_max], 'b--');
    l3 = area(edges(1:end-1), N_grad);
    l3.FaceColor = 'r';
    l3.FaceAlpha = 0.5;
    l3.LineWidth = 2.0;
    l4 = plot(rms_grad_all * [1, 1], [0, y_max], 'r--');
    
    xlim([0 x_max])
    ylim([0 y_max])

    legend({'\sigma_{X_0}, Ref.', 'RMS \sigma_{X_0}, Ref.', '\sigma_{X_0}, Grad.', 'RMS \sigma_{X_0}, Grad.'}, 'fontsize', 10)
    xlabel('\sigma_X (pix.)')
    ylabel('PDF')
    title('Position Uncertainty');
    
    set(gcf, 'Position', [360   584   441   316])
    
    if save_figures
        save_figure_to_png_eps_fig(figure_save_directory, 'uncertainty-hist-ref-vs-grad', [1, 0, 0]);
    end
    
end
