% script to evaulate the cramer-rao lower bound for bos images of
% spark induced flow

clear
close all
clc

rmpath('/scratch/shannon/a/lrajendr/Software/prana');
restoredefaultpath;
% addpath ../prana-master/
% addpath ../error-analysis-codes/helper-codes/mCodes/
% addpath ../error-analysis-codes/post-processing-codes/
% addpath ../error-analysis-codes/ptv-codes/

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
image_read_list = [2, 10]; %, 40];
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
dot_diameter = 4;
% minimum expected area for a group of pixels to be identified as a dot
% (pix.^2)
min_area = 9; %dot_diameter^2 * 0.5;
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

% minimum allowable position uncertainty [pix.]
min_uncertainty_threshold = 1e-3;
% maximum allowable position uncertainty [pix.]
max_uncertainty_threshold = 0.2;
% bins for position uncertainty histogram
edges = linspace(0, max_uncertainty_threshold, 100);

%% load image mask
if image_masking
    % load image mask
    mask = imread(fullfile(image_mask_directory, image_mask_filename));
    % flip the image mask upside down and convert to double data type
    mask = double(flipud(mask));
    % set values of the mask greater than 0 (= 255) to be 1
    mask(mask > 0) = 1;

    % identify extents of the masked region
    stats = regionprops(mask, 'boundingbox');
    xmin_mask = stats.BoundingBox(1);
    ymin_mask = stats.BoundingBox(2);
    xmax_mask = xmin_mask + stats.BoundingBox(3);
    ymax_mask = ymin_mask + stats.BoundingBox(4);
end

%% calculate reference dot location from dot positions

% load camera mapping coefficients
mapping_coefficients = load(fullfile(calibration_directory, ['camera_model_type=' num2str(order_z) '.mat']));

% load calibration results
calibration_results = load(fullfile(calibration_directory, 'calibration_data.mat'));

% -----------------------------------------
% assign values to the parameter structure
% -----------------------------------------
% extract x and y co-ordinates of the origin marker in the calibration
% image (um)
experimental_parameters.bos_pattern.X_Min = min(calibration_results.calibration_data.x_world_full{1}) * 1e3;
experimental_parameters.bos_pattern.Y_Min = min(calibration_results.calibration_data.y_world_full{1}) * 1e3;
% spacing between dots (um)
experimental_parameters.bos_pattern.dot_spacing = 2 * dot_size * 1e3; 
% pixel pitch (um)
experimental_parameters.camera_design.pixel_pitch = pixel_pitch;
% magnification (unitless)
experimental_parameters.lens_design.magnification = 0.83; %experimental_parameters.camera_design.pixel_pitch/magnification; %6.45/33;
% number of pixels on the camera sensor
experimental_parameters.camera_design.y_pixel_number = y_pixel_number;
experimental_parameters.camera_design.x_pixel_number = x_pixel_number;
% expected field of view given the dot spacing, magnification and the
% number of pixels on the camera sensor
field_of_view = experimental_parameters.camera_design.x_pixel_number * experimental_parameters.camera_design.pixel_pitch / experimental_parameters.lens_design.magnification;
% expected number of dots based on the field of view
experimental_parameters.bos_pattern.dot_number = round(field_of_view/experimental_parameters.bos_pattern.dot_spacing) * 1; %* 2;

% ------------------------------------------------
% calculate expected dot locations on image plane
% ------------------------------------------------
% create a grid of co-ordinates expected to align with the dots on the
% object plane
x_array = experimental_parameters.bos_pattern.X_Min + experimental_parameters.bos_pattern.dot_spacing * (0:experimental_parameters.bos_pattern.dot_number);
y_array = experimental_parameters.bos_pattern.Y_Min + experimental_parameters.bos_pattern.dot_spacing * (0:experimental_parameters.bos_pattern.dot_number);
% combine the two 1D arrays into a 2D grid
[positions.x, positions.y] = meshgrid(x_array, y_array);

% calculate reference dot locations on the image plane based on the
% expected locations on the object plane and the camera mapping function.
fprintf('calculating reference dot locations\n');
[pos_ref_dots.x, pos_ref_dots.y] = calculate_reference_dot_locations_new(positions, experimental_parameters, camera_model, mapping_coefficients, order_z, starting_index_x, starting_index_y);
% flip the y co-ordinates of the dots to account for the image being
% flipped upside down
pos_ref_dots.y = experimental_parameters.camera_design.y_pixel_number - pos_ref_dots.y;

% remove points outside FOV
indices = pos_ref_dots.x < 1 | pos_ref_dots.x > experimental_parameters.camera_design.x_pixel_number-1 | pos_ref_dots.y < 1 | pos_ref_dots.y > experimental_parameters.camera_design.y_pixel_number-1;
pos_ref_dots.x(indices) = [];
pos_ref_dots.y(indices) = [];
num_dots_ref = numel(pos_ref_dots.x); % - sum(indices);

% remove dots in masked regions. first set dots in masked region to be NaN
for dot_index = 1:num_dots_ref
    if round(pos_ref_dots.x(dot_index)) > size(mask, 2) || ...
            round(pos_ref_dots.y(dot_index)) > size(mask, 1) 
        pos_ref_dots.x(dot_index) = NaN;
        pos_ref_dots.y(dot_index) = NaN;
        continue;
    end        
    if mask(round(pos_ref_dots.y(dot_index)), round(pos_ref_dots.x(dot_index))) == 0
        pos_ref_dots.x(dot_index) = NaN;
        pos_ref_dots.y(dot_index) = NaN;
    end
end

% find and remove NaN indices
nan_indices = isnan(pos_ref_dots.x) | isnan(pos_ref_dots.y);
pos_ref_dots.x(nan_indices) = [];
pos_ref_dots.y(nan_indices) = [];

%% load background image for subtraction

if background_subtraction
    im_bg = imread(fullfile(background_image_directory, background_image_filename));
    % estimate size of background image
    NR_bg = size(im_bg, 1);
    NC_bg = size(im_bg, 2);    
    % pad background image at the top to ensure it is the same size as the
    % images to be processed
    im_bg = padarray(im_bg, [y_pixel_number-NR_bg x_pixel_number-NC_bg], 0, 'pre');
    % flip image upside down to align y-axes
    im_bg = double(flipud(im_bg));
    % mask image if required
    if image_masking
        im_bg = im_bg .* mask;
    end
end

%% load image filenames

% load list of runs for the current pressure condition
[runs, num_runs] = get_directory_listing(fullfile(top_image_directory, pulse_parameter_name), 'test*');
% run index
test_index = 2;

% image directory for this case
current_image_directory = fullfile(runs(test_index).folder, runs(test_index).name);
% obtain list of files in the reference image directory
[ref_images, ~] = get_directory_listing(fullfile(current_image_directory, 'ref'), 'im*.tif');
% obtain list of files in the gradient image directory
[grad_images, ~] = get_directory_listing(fullfile(current_image_directory, 'grad'), 'im*.tif');

% directory to store results for the current case
current_results_directory = fullfile(top_results_directory, pulse_parameter_name, runs(test_index).name, ['bg_subtraction=' logical_string{background_subtraction + 1} '_median_filtering=' logical_string{median_filtering+1} '_min_subtraction=' logical_string{minimum_subtraction + 1} '_subpixel_fit=' subpixel_fit '_default_iwc=' logical_string{default_iwc + 1} '_fixed_ref_locxy_ell']);
if ~exist(current_results_directory, 'dir')
    mkdir(current_results_directory);
end

%% identify dots on the images

fprintf('identifying dots in images\n');

% --------------------------
% Reference Image
% --------------------------
fprintf('Reference Image\n');
% load image
im_ref = imread(fullfile(ref_images(1).folder, ref_images(1).name));
% flip image upside down so bottom pixel is y = 1
im_ref = flipud(im_ref);

% mask image if required
if image_masking
    im_ref = double(im_ref) .* mask;
end

% subtract background if required
if background_subtraction
    im_ref = im_ref - im_bg;
    im_ref(im_ref < 0) = 0;
end

% perform minimum subtraction if required
if minimum_subtraction
    im_ref = im_ref - minimum_subtraction_intensity_level;
    im_ref(im_ref < 0) = 0;
end

% declare cells and arrays to hold co-ordinates of identified dots
SIZE1_ref = cell(1, num_ref_images);
X_ref = cell(1, num_ref_images);
Y_ref = cell(1, num_ref_images);
Z_ref = cell(1, num_ref_images);
d_x_ref = cell(1, num_ref_images);
d_y_ref = cell(1, num_ref_images);
R_ref = cell(1, num_ref_images);
I_ref = cell(1, num_ref_images);
d_avg_ref = cell(1, num_ref_images);


%% run both ID and sizing for first reference image

image_index = 1;

[SIZE1_ref{image_index}.XYDiameter, SIZE1_ref{image_index}.peaks, SIZE1_ref{image_index}.mapsizeinfo, SIZE1_ref{image_index}.locxy, SIZE1_ref{image_index}.mapint] = combined_ID_size_apriori_10(im_ref, pos_ref_dots.x, pos_ref_dots.y, dot_diameter, subpixel_fit, default_iwc, min_area, W_area, W_intensity, W_distance);
% extract results
num_p = size(SIZE1_ref{image_index}.XYDiameter, 1);
locxy = SIZE1_ref{image_index}.locxy;
mapsizeinfo = SIZE1_ref{image_index}.mapsizeinfo;

%% identify dots on other reference images

for image_index = 2:num_ref_images    
    fprintf('image: %d\n', image_index);
    
    % load image
    im = imread(fullfile(ref_images(image_index).folder, ref_images(image_index).name));    
    % flip image upside down so bottom pixel is y = 1
    im = double(flipud(im));
    % mask image if required
    if image_masking
        im = double(im) .* mask;
    end
    % subtract background if required
    if background_subtraction
        im = im - im_bg;
    end
    
    if minimum_subtraction
        im = im - minimum_subtraction_intensity_level;
        indices = im < 0;
        im(indices) = 0;
    end
    % identify dots
    % for other reference images, extract neighborhood from the first image and run sizing on
    % that map
    SIZE1_ref{image_index}.locxy = SIZE1_ref{1}.locxy;
    SIZE1_ref{image_index}.mapint = cell(1, num_p);
    % loop through all dots and extract intensity maps from the neighborhood
    for p = 1:num_p
        rmin = locxy(p,2);
        cmin = locxy(p,1);
        rmax = rmin + mapsizeinfo(p,1) - 1;
        cmax = cmin + mapsizeinfo(p,2) - 1;

        % if the dot had failed identification, then skip
        if nnz(isnan([rmin, cmin, rmax, cmax])) < 1
            SIZE1_ref{image_index}.mapint{p} = im(rmin:rmax, cmin:cmax);
        else
            SIZE1_ref{image_index}.mapint{p} = SIZE1_ref{1}.mapint{p};
        end
    end

    SIZE1_ref{image_index}.XYDiameter = perform_dot_sizing(num_p, SIZE1_ref{image_index}.mapint, locxy, subpixel_fit, default_iwc);

    % extract properties of identified dots
    X_ref{image_index} = SIZE1_ref{image_index}.XYDiameter(:,1);
    Y_ref{image_index} = SIZE1_ref{image_index}.XYDiameter(:,2);
    Z_ref{image_index} = zeros(size(X_ref{image_index}));
    d_x_ref{image_index} = SIZE1_ref{image_index}.XYDiameter(:,3);
    d_y_ref{image_index} = SIZE1_ref{image_index}.XYDiameter(:,4);
    d_avg_ref{image_index} = sqrt(SIZE1_ref{image_index}.XYDiameter(:,3).^2 + SIZE1_ref{image_index}.XYDiameter(:,4).^2);
    R_ref{image_index} = SIZE1_ref{image_index}.XYDiameter(:,5);
    I_ref{image_index} = SIZE1_ref{image_index}.XYDiameter(:,6);
end

%% calculate dot properties and statistics across time series

% ---------------------------------------------
% find matching dots between successive frames
% ---------------------------------------------
fprintf('finding matching dots across frames \n');
% reference images
[X_all_ref, Y_all_ref, Z_all_ref, d_x_all_ref, d_y_all_ref, R_all_ref, I_all_ref] = find_matching_dots_across_time_series_02(X_ref, Y_ref, Z_ref, d_x_ref, d_y_ref, R_ref, I_ref);

%% calculate statistics

% ---------------------------------------------
% calculate mean properties
% ---------------------------------------------
fprintf('calculating mean properties \n');
% reference image
[X_mean_ref, Y_mean_ref, Z_mean_ref, d_x_mean_ref, d_y_mean_ref, R_mean_ref, I_mean_ref] = calculate_mean_across_time_series_02(X_all_ref, Y_all_ref, Z_all_ref, d_x_all_ref, d_y_all_ref, R_all_ref, I_all_ref);
d_avg_mean_ref = sqrt(d_x_mean_ref.^2 + d_y_mean_ref.^2);

% ---------------------------------------------
% calculate median properties
% ---------------------------------------------
fprintf('calculating median properties \n');
% reference image
[X_median_ref, Y_median_ref, Z_median_ref, d_x_median_ref, d_y_median_ref, R_median_ref, I_median_ref] = calculate_median_across_time_series_02(X_all_ref, Y_all_ref, Z_all_ref, d_x_all_ref, d_y_all_ref, R_all_ref, I_all_ref);
d_avg_median_ref = sqrt(d_x_median_ref.^2 + d_y_median_ref.^2);

% -----------------------------------------------------------
% calculate std properties for the matched dots across frames 
% -----------------------------------------------------------
fprintf('calculating std properties \n');
% reference image
[X_std_ref, Y_std_ref, Z_std_ref, d_x_std_ref, d_y_std_ref, R_std_ref, I_std_ref] = calculate_std_across_time_series_02(X_all_ref, Y_all_ref, Z_all_ref, d_x_all_ref, d_y_all_ref, R_all_ref, I_all_ref);

% -----------------------------------------------------------
% calculate interquartile (84-16) range for the matched dots across frames 
% -----------------------------------------------------------
fprintf('calculating per properties \n');
% reference image
[X_per_ref, Y_per_ref, Z_per_ref, d_x_per_ref, d_y_per_ref, R_per_ref, I_per_ref] = calculate_per_across_time_series_02(X_all_ref, Y_all_ref, Z_all_ref, d_x_all_ref, d_y_all_ref, R_all_ref, I_all_ref);

%% identify dots on gradient image
% --------------------------
% Gradient Image
% --------------------------
fprintf('Gradient Images\n');
% number of images to be read
num_images_read = numel(image_read_list);

% declare cells and arrays to hold co-ordinates of identified dots
SIZE1_grad = cell(1, num_images_read);
X_grad = cell(1, num_images_read);
Y_grad = cell(1, num_images_read);
Z_grad = cell(1, num_images_read);
d_x_grad = cell(1, num_images_read);
d_y_grad = cell(1, num_images_read);
d_avg_grad = cell(1, num_images_read);
R_grad = cell(1, num_images_read);
I_grad = cell(1, num_images_read);

% loop through all images
for image_index = image_read_list

    fprintf('image: %d\n', image_index);

    %% create directory to save workspace for the current case
    
    workspace_save_directory = fullfile(current_results_directory, ['im' num2str(image_index, '%04d')], 'workspace');
    mkdir_c(workspace_save_directory);    

    %% load and process image
    
    % load image
    im = imread(fullfile(grad_images(image_index).folder, grad_images(image_index).name));    
    % flip image upside down so bottom pixel is y = 1
    im = double(flipud(im));
    
    % mask image if required
    if image_masking
        im = double(im) .* mask;
    end
    
    % subtract background if required
    if background_subtraction
        im = im - im_bg;
    end
    
    if minimum_subtraction
        im = im - minimum_subtraction_intensity_level;
        indices = im < 0;
        im(indices) = 0;
    end
    % identify dots
    [SIZE1_grad{image_index}.XYDiameter, SIZE1_grad{image_index}.peaks, SIZE1_grad{image_index}.mapsizeinfo, SIZE1_grad{image_index}.locxy, SIZE1_grad{image_index}.mapint]=combined_ID_size_apriori_10(im, pos_ref_dots.x, pos_ref_dots.y, dot_diameter, subpixel_fit, default_iwc, min_area, W_area, W_intensity, W_distance);

    % extract properties of identified dots
    X_grad{image_index} = SIZE1_grad{image_index}.XYDiameter(:,1);
    Y_grad{image_index} = SIZE1_grad{image_index}.XYDiameter(:,2);
    Z_grad{image_index} = zeros(size(X_grad{image_index}));
    d_x_grad{image_index} = SIZE1_grad{image_index}.XYDiameter(:,3);
    d_y_grad{image_index} = SIZE1_grad{image_index}.XYDiameter(:,4);
    d_avg_grad{image_index} = sqrt(d_x_grad{image_index}.^2 + d_y_grad{image_index}.^2);
    R_grad{image_index} = SIZE1_grad{image_index}.XYDiameter(:,5);
    I_grad{image_index} = SIZE1_grad{image_index}.XYDiameter(:,6);

    %% dot tracking

    % ------------------------------------------------------------------
    % find matching dots between reference and the first gradient image
    % ------------------------------------------------------------------
    s_radius = 3;
    fprintf('tracking dot locations between reference image and first grad image\n');

    % run tracking
%     [tracks]=weighted_nearest_neighbor3D(X_grad{image_index}, X_ref{1}, X_grad{image_index}, Y_grad{image_index}, Y_ref{1}, Y_grad{image_index},...
%         Z_grad{image_index}, Z_ref{1}, Z_grad{image_index}, d_avg_grad{image_index}, d_avg_ref{1}, I_grad{image_index}, I_ref{1}, weights, s_radius);
    [tracks]=weighted_nearest_neighbor3D(X_grad{image_index}, X_median_ref, X_grad{image_index}, Y_grad{image_index}, Y_median_ref, Y_grad{image_index},...
        Z_grad{image_index}, Z_median_ref, Z_grad{image_index}, d_avg_grad{image_index}, d_avg_median_ref, I_grad{image_index}, I_median_ref, weights, s_radius);

    % perform correlation correction
    
    fprintf('correlation correction \n');
    % get final sub-pixel displacement estimate by cross-correlation intensity
    % maps of the dots
    num_tracks = size(tracks, 1);
    U = zeros(num_tracks,1);
    V = zeros(num_tracks,1);
    for track_index = 1:num_tracks
        if rem(track_index, 1000) == 0
            fprintf('Track: %d of %d\n', track_index, num_tracks);
        end
        % extract current track
        track_current = tracks(track_index, :);
        [U(track_index), V(track_index)] = cross_correlate_dots_06(im, im_ref, SIZE1_grad{image_index}, SIZE1_ref{1}, track_current, 'dcc', 'lsg', true, true);            
    end
    
    % append results to track
    tracks = [tracks, U, V];

    %% calculate amplification ratio and position uncertainty from model

    fprintf('calculating crlb \n');

    % number of dots for which the crlb needs to be calculated
    num_tracks = size(tracks,1);

    % declare arrays to hold results
    X_ref_tracked = nans(num_tracks, 1);
    Y_ref_tracked = nans(num_tracks, 1);
    
    X_grad_tracked = nans(num_tracks, 1);
    Y_grad_tracked = nans(num_tracks, 1);

    X_std_ref_tracked = nans(num_tracks, 1);
    Y_std_ref_tracked = nans(num_tracks, 1);

    X_std_grad_tracked = nans(num_tracks, 1);
    Y_std_grad_tracked = nans(num_tracks, 1);

    AR_x_tracked = nans(num_tracks, 1);
    AR_y_tracked = nans(num_tracks, 1);

    % loop through all dots and estimate crlb
    for track_index = 1:num_tracks
        % display progress to user
        
        if rem(track_index, 1000) == 0
            fprintf('Track number: %d\n', track_index);
        end        
        
        %% extract dot index
        
        % reference
        p_ref = tracks(track_index, 12);        
        % gradient
        p_grad = tracks(track_index, 11);
        
        %% extract dot image co-ordinates
        
        % reference
        X_ref_tracked(track_index) = X_median_ref(p_ref);
        Y_ref_tracked(track_index) = Y_median_ref(p_ref);

        % gradient
        X_grad_tracked(track_index) = X_grad{image_index}(p_grad);
        Y_grad_tracked(track_index) = Y_grad{image_index}(p_grad);
        
        %% calculate amplification ratio
        
        [AR_x_tracked(track_index), AR_y_tracked(track_index)] = calculate_amplification_ratio(d_x_median_ref(p_ref), d_y_median_ref(p_ref), R_median_ref(p_ref), I_median_ref(p_ref), ...
                                                    d_x_grad{image_index}(p_grad), d_y_grad{image_index}(p_grad), R_grad{image_index}(p_grad), I_grad{image_index}(p_grad));
        
        %% calculate position uncertainty
        
        % reference
        X_std_ref_tracked(track_index) = X_std_ref(p_ref);
        Y_std_ref_tracked(track_index) = Y_std_ref(p_ref);
        
        % gradient
        X_std_grad_tracked(track_index) = X_std_ref(p_ref) * AR_x_tracked(track_index);
        Y_std_grad_tracked(track_index) = Y_std_ref(p_ref) * AR_y_tracked(track_index);
    end
    
    % retain only finite values
    AR_x_tracked(~isfinite(AR_x_tracked)) = 1;
    AR_y_tracked(~isfinite(AR_y_tracked)) = 1;

    %% calculate pdf of position uncertainty
    
    % identify measurements that have displacements greater than the 10th
    % percentile
    indices = abs(U) > prctile(abs(U), 25) & abs(V) > prctile(abs(V), 25);

    % aggregate measuremnts
    X_std_ref_tracked_all = [X_std_ref_tracked(indices); Y_std_ref_tracked(indices)];
    X_std_grad_tracked_all = [X_std_grad_tracked(indices); Y_std_grad_tracked(indices)];
    
    % remove invalid measurements
    X_std_ref_tracked_all(abs(X_std_ref_tracked_all) < min_uncertainty_threshold | ... 
        abs(X_std_ref_tracked_all) > max_uncertainty_threshold) = [];    
    X_std_grad_tracked_all(abs(X_std_grad_tracked_all) < min_uncertainty_threshold | ...
        abs(X_std_grad_tracked_all) > max_uncertainty_threshold) = [];
        
    % calculate pdf of position uncertainty
    [N_ref, ~] = histcounts(X_std_ref_tracked_all, edges, 'Normalization', 'pdf');
    [N_grad, ~] = histcounts(X_std_grad_tracked_all, edges, 'Normalization', 'pdf');
    
    % calculate rms of uncertainty
    rms_ref_all = rms(X_std_ref_tracked_all(:), 'omitnan');
    rms_grad_all = rms(X_std_grad_tracked_all(:), 'omitnan');
    
    %% close all figures
    
    close all;

    %% save workspace to file
    
    save(fullfile(workspace_save_directory, [extract_script_name(mfilename('fullpath')) '.mat']));

end
