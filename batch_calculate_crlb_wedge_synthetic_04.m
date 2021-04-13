% script to evaulate the cramer-rao lower bound for bos images of
% supersonic flow over a wedge and compare theoretical estimates to
% experiment.

clear
close all
clc

rmpath('/scratch/shannon/a/lrajendr/Software/prana');
restoredefaultpath;
% addpath ../prana-master/
% addpath /scratch/shannon/c/aether/Projects/BOS/error-analysis/analysis/src/error-analysis-codes/helper-codes/mCodes/
% addpath /scratch/shannon/c/aether/Projects/BOS/error-analysis/analysis/src/error-analysis-codes/post-processing-codes/
% addpath ../error-analysis-codes/ptv-codes/
addpath(genpath('/scratch/shannon/c/aether/Projects/BOS/general-codes/matlab-codes/'));
% addpath /scratch/shannon/a/lrajendr/Software/general-mCodes/
addpath /scratch/shannon/a/lrajendr/Software/cbrewer/
addpath ../dot-tracking-package/
% addpath 
dbstop if error
logical_string = {'False'; 'True'};

%% experiment settings

% --------------------------
% wind tunnel settings
% --------------------------
% chamber pressure (psig) - from read out near tunnel
pressure = 55;
% stagnation temperature (deg. C)
T0 = 20.5;
% atmospheric pressure (psi)
p_a = 14.7;
% psi to pascal conversion
psi_to_pa = 6894.76;
% gas constant for air (J/kg-K)
R = 287;
% specific heat ratio
gamm = 1.4;
% upstream mach number
M1 = 2.56; %2.4;
% upstream stagnation density (kg/m^3)
% rho_01 = 3 * 1.225;
rho_01 = (pressure + p_a) * psi_to_pa/(R * (273.15 + T0));

% --------------------------
% wedge settings
% --------------------------

% location of wedge tip (from looking at the correlation vector field)
x0 = 865; %897; %846; %1066; %830; %1392 - 830;
y0 = 631; %406; %421; %396; %492;
% location of wedge shoulder (pix.)
xs = 360;
ys = 545; %480;
% angle of the wedge shoulder (deg.)
shoulder_angle = 0;
% wedge half angle (deg.)
wedge_angle = 10; %1/5;

% --------------------------
% dot pattern settings
% --------------------------
% geoemtric dot diameter (um)
d_geo = 0;
% dot size (pix.)
dot_size_pixels = 3;
% expected dot spacing in the image plane (pix.)
dot_spacing = 6;
% % distance between dot pattern and density gradient field (m)
% Z_D = 1 * 0.0254;
% % thickness of the density gradient field/wind-tunnel (m)
% Z_g = 1 * 0.0254;

% --------------------------
% imaging settings
% --------------------------
% lens model
lens_model = 'thin-lens'; %'apparent';
% magnification (um/pix.)
magnification = 10^4/180; %33;
% number of pixels on the camera sensor
y_pixel_number = 512;
x_pixel_number = 1024;
% size of a pixel on the camera sensor (um)
pixel_pitch = 20.8;
% f-number of the camera aperture
f_number = 22;
% focal length of the camera lens (mm)
focal_length = 105;
% image noise level (%)
noise_gray_level = 0.05;

% --------------------------
% density gradient settings
% --------------------------
% algorithm used for ray tracing
ray_tracing_algorithm = 'rk4';
% distance between dot pattern and density gradient field (mm)
Z_D = 1 * 0.0254 * 1e3;
% thickness of the density gradient field/wind-tunnel (mm)
Z_g = 1 * 0.0254 * 1e3;
% distance between density gradietns and camera lens (mm)
Z_A = (10.5*0.0254*1e3 + Z_g/2);
% distance between dot pattern and camera lens (mm)
z_obj = Z_A + Z_D;
% distance between dot pattern and far end of density field (mm)
zmax = Z_D + Z_g/2.0;
% distance between dot pattern and near end of density field (mm)
zmin = Z_D - Z_g/2.0;
% number of grid points along x
nx = 234; %700; %113;
% number of grid points along y
ny = 84; %251; %41;
% number of grid points along z
nz = 51;

%% processing settings
% ------------------------
% read/write settings
% ------------------------
% top level directory containing images
top_image_directory = fullfile('/scratch/shannon/c/aether/Projects/BOS/crlb/analysis/data/images/synthetic/wedge/regular-overlap=False/', lens_model, [num2str(x_pixel_number, '%d') 'x' num2str(y_pixel_number, '%d')], ...
    ray_tracing_algorithm, ['zobj' num2str(floor(z_obj), '%d') '-zmin' num2str(floor(zmin), '%d') '-zmax' num2str(floor(zmax), '%d') '-nx' num2str(nx, '%d') '-ny' num2str(ny, '%d') '-nz' num2str(nz, '%d')], ...
    ['f=' num2str(focal_length, '%d') 'mm_f' num2str(f_number, '%02d') '_l_p=' num2str(floor(pixel_pitch), '%02d') 'um'], ['dg=' num2str(d_geo, '%d') 'um_dp=' num2str(dot_size_pixels, '%.2f') 'pix_spacing=' num2str(dot_spacing, '%.2f') 'pix'], 'noise00'); 
% top level directory to store the results
top_results_directory = fullfile('/scratch/shannon/c/aether/Projects/BOS/crlb/analysis/results/synthetic/wedge/regular-overlap=False/', lens_model, [num2str(x_pixel_number, '%d') 'x' num2str(y_pixel_number, '%d')], ...
    ray_tracing_algorithm, ['zobj' num2str(floor(z_obj), '%d') '-zmin' num2str(floor(zmin), '%d') '-zmax' num2str(floor(zmax), '%d') '-nx' num2str(nx, '%d') '-ny' num2str(ny, '%d') '-nz' num2str(nz, '%d')], ...
    ['f=' num2str(focal_length, '%d') 'mm_f' num2str(f_number, '%02d') '_l_p=' num2str(floor(pixel_pitch), '%02d') 'um'], ['dg=' num2str(d_geo, '%d') 'um_dp=' num2str(dot_size_pixels, '%.2f') 'pix_spacing=' num2str(dot_spacing, '%.2f') 'pix'], ['noise' num2str(floor(noise_gray_level*1000), '%03d')]);

% ----------------------------
% general processing settings
% ----------------------------
% perform background subtraction? (true/false)
background_subtraction = false;
% peform image masking? (true/false)
image_masking = false;
% perform median filtering of dot positions across time series before
% calculating position estimation variance? (true/false)
median_filtering = false;
% number of trials to calculate position estimation variance
num_trials = 1e3;
% buffer to remove points close to boundaries
left_boundary_buffer = 200;
right_boundary_buffer = 200;
top_boundary_buffer = 100;
bottom_boundary_buffer = 100;

% ------------------------
% calibration settings
% ------------------------
% directory containing calibration images
calibration_directory = fullfile(top_image_directory, 'calibration-crlb');
% camera model
camera_model = 'thin-lens';
% order of the z term in the polynomial mapping function (x and y are
% cubic).
order_z = 1;
% offset in the co-ordinate system of the sensor for synthetic images
% (pix.)
starting_index_x = -1; %0;
starting_index_y = 0; %1;

% ------------------------
% identification settings
% ------------------------
% expected dot diameter in the image plane (pix.)
dot_diameter = dot_size_pixels; % 1.5 * dot_size_pixels;
% minimum expected area for a group of pixels to be identified as a dot
% (pix.^2)
min_area = 9; %dot_diameter^2 * 0.5;
% subpixel fit to used for centroid estimation
subpixel_fit = 'clsg';
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

% -------------------
% plot settings
% -------------------
% save figures? (true/false)
save_figures = true;
% plot error histograms for the position estimation variance calculation?
% (true/false)
plot_error_histogram = false;
% min color level for position standard deviation (pix.)
global_cmin = 0.02; %1;
% max color level for position standard deviation (pix.)
global_cmax = 0.05; %7;
% factor to scale the displacements for quiver plots 
scale_factor = 0.05;
% extract line colors
colors = lines(3);

%% directory info

% directory containing for the current case
current_image_directory = fullfile(top_image_directory, '1');
% directory to store results for the current case
current_results_directory = fullfile(top_results_directory, ['bg_subtraction=' logical_string{background_subtraction + 1} '_median_filtering=' logical_string{median_filtering+1}], [subpixel_fit '-ell']);
mkdir_c(current_results_directory);

% directory to store figures for the current case
figure_save_directory = fullfile(current_results_directory, 'figures');
mkdir_c(figure_save_directory);

%% calculate reference dot location from dot positions

% load parameters for a sample image
image_generation_parameters = load(fullfile(top_image_directory, '1', 'parameters.mat'));

% load dot positions for a sample image
positions = load(fullfile(top_image_directory, '1', 'positions.mat'));

% calculate reference dot locations on the image plane based on the
% expected locations on the object plane and the camera mapping function.
fprintf('calculating reference dot locations\n');
pos_ref_dots.x = positions.x;
pos_ref_dots.y = positions.y;

[pos_ref_dots.x, pos_ref_dots.y] = calculate_reference_dot_locations_new(positions, image_generation_parameters, camera_model, [], [], starting_index_x, starting_index_y);
% flip the y co-ordinates of the dots to account for the image being
% flipped upside down
pos_ref_dots.y = image_generation_parameters.camera_design.y_pixel_number - pos_ref_dots.y;

% remove points outside FOV
% indices = pos_ref_dots.x < 1 | pos_ref_dots.x > image_generation_parameters.camera_design.x_pixel_number-1 | pos_ref_dots.y < 1 | pos_ref_dots.y > image_generation_parameters.camera_design.y_pixel_number-1;
indices = pos_ref_dots.x < left_boundary_buffer | ...
    pos_ref_dots.x > image_generation_parameters.camera_design.x_pixel_number - right_boundary_buffer | ...
    pos_ref_dots.y < top_boundary_buffer | ...
    pos_ref_dots.y > image_generation_parameters.camera_design.y_pixel_number - bottom_boundary_buffer;
pos_ref_dots.x(indices) = [];
pos_ref_dots.y(indices) = [];
num_dots_ref = numel(pos_ref_dots.x); % - sum(indices);

% remove dots in masked regions. first set dots in masked region to be NaN
if image_masking
    for dot_index = 1:num_dots_ref
        if mask(round(pos_ref_dots.y(dot_index)), round(pos_ref_dots.x(dot_index))) == 0
            pos_ref_dots.x(dot_index) = NaN;
            pos_ref_dots.y(dot_index) = NaN;
        end
    end
    % find and remove NaN indices
    nan_indices = isnan(pos_ref_dots.x) | isnan(pos_ref_dots.y);
    pos_ref_dots.x(nan_indices) = [];
    pos_ref_dots.y(nan_indices) = [];    
end

%% calculate dot positions from light rays

% load light ray data
ray_data = load_lightray_data_03(current_image_directory);

% extract final positions and directions of light rays for each dot
% (pix.)
[x1, y1] = pos_to_xy_pix_04(ray_data{1}.pos.x, ray_data{1}.pos.y, image_generation_parameters, -1, -0.5);
[x2, y2] = pos_to_xy_pix_04(ray_data{2}.pos.x, ray_data{2}.pos.y, image_generation_parameters, -1, -0.5);

[pos_ref_rays_1.x, pos_ref_rays_1.y] = calculate_centroids_from_lightrays_06(x1, y1, image_generation_parameters.bos_pattern.particle_number_per_grid_point * image_generation_parameters.bos_pattern.lightray_number_per_particle);
[pos_ref_rays_2.x, pos_ref_rays_2.y] = calculate_centroids_from_lightrays_06(x2, y2, image_generation_parameters.bos_pattern.particle_number_per_grid_point * image_generation_parameters.bos_pattern.lightray_number_per_particle);

% remove nan values
nan_indices = isnan(pos_ref_rays_1.x) | isnan(pos_ref_rays_1.y) | isnan(pos_ref_rays_2.x) | isnan(pos_ref_rays_2.y);
pos_ref_rays_1.x(nan_indices) = [];
pos_ref_rays_1.y(nan_indices) = [];
pos_ref_rays_2.x(nan_indices) = [];
pos_ref_rays_2.y(nan_indices) = [];

z_rays_ref = zeros(size(pos_ref_rays_1.x));
z_rays_grad = zeros(size(pos_ref_rays_2.x));

d_rays_ref = zeros(size(pos_ref_rays_1.x));
d_rays_grad = zeros(size(pos_ref_rays_2.x));

I_rays_ref = zeros(size(pos_ref_rays_1.x));
I_rays_grad = zeros(size(pos_ref_rays_2.x));

% run tracking
fprintf('Tracking light rays from reference to grad images\n');
[tracks_rays]=weighted_nearest_neighbor3D(pos_ref_rays_2.x, pos_ref_rays_1.x, pos_ref_rays_2.x, pos_ref_rays_2.y, pos_ref_rays_1.y, pos_ref_rays_2.y,...
    z_rays_grad, z_rays_ref, z_rays_grad, d_rays_grad, d_rays_ref, I_rays_grad, I_rays_ref, weights, s_radius);

x_track_rays = tracks_rays(:,1);
y_track_rays = tracks_rays(:,3);
u_track_rays = tracks_rays(:,2) - tracks_rays(:,1);
v_track_rays = tracks_rays(:,4) - tracks_rays(:,3);

% remove boundary tracks
[x_track_rays, y_track_rays, u_track_rays, v_track_rays] = remove_boundary_points_scattered(x_track_rays, y_track_rays, u_track_rays, v_track_rays, image_generation_parameters.camera_design.x_pixel_number, image_generation_parameters.camera_design.y_pixel_number, left_boundary_buffer, right_boundary_buffer, bottom_boundary_buffer, top_boundary_buffer);
% plot tracks
figure
quiver(x_track_rays, y_track_rays, u_track_rays/scale_factor, v_track_rays/scale_factor, 'AutoScale', 'off') 
annotate_image(gcf, gca);
title('Tracked Rays')
if save_figures
    saveas(gcf, fullfile(figure_save_directory, 'track-rays.png'), 'png');
    saveas(gcf, fullfile(figure_save_directory, 'track-rays.eps'), 'epsc');    
end

%% identify dots on the images

fprintf('identifying dots in images\n');

% --------------------------
% Reference Image
% --------------------------
fprintf('Reference Image\n');
% load image
% im_ref = imread(fullfile(current_image_directory, 'bos_pattern_image_1.tif'));
fid = fopen(fullfile(current_image_directory, 'bos_pattern_image_1.bin'));
a = fread(fid, 'single');
fclose(fid);
im_ref = reshape(a, image_generation_parameters.camera_design.x_pixel_number, image_generation_parameters.camera_design.y_pixel_number);
im_ref = im_ref';

% % flip image upside down so bottom pixel is y = 1
% im_ref = flipud(im_ref);
% flip image left to right
im_ref = fliplr(im_ref);
% mask image if required
if image_masking
    im_ref = double(im_ref) .* mask;
end
% subtract background if required
if background_subtraction
    im_ref = im_ref - im_bg;
end

% set image pixels outside buffer to zero
im_ref(:, 1:left_boundary_buffer) = 0;
im_ref(:, image_generation_parameters.camera_design.x_pixel_number - right_boundary_buffer:end) = 0;
im_ref(1:bottom_boundary_buffer, :) = 0;
im_ref(image_generation_parameters.camera_design.y_pixel_number - top_boundary_buffer:end, :) = 0;

% identify dots on the reference image
[SIZE_ref.XYDiameter, SIZE_ref.peaks, SIZE_ref.mapsizeinfo, SIZE_ref.locxy, SIZE_ref.mapint] = combined_ID_size_apriori_10(im_ref, pos_ref_dots.x, pos_ref_dots.y, dot_diameter+2, subpixel_fit, default_iwc, min_area, W_area, W_intensity, W_distance);

% extract dot co-ordinates from sizing results
[X_ref, Y_ref, Z_ref, d_x_ref, d_y_ref, R_ref, I_ref] = extract_dot_properties_02(SIZE_ref.XYDiameter);

% remove dots outside ROI
[X_ref, Y_ref, Z_ref, d_x_ref, d_y_ref, R_ref, I_ref] = nan_dots_outside_ROI_02(X_ref, Y_ref, Z_ref, d_x_ref, d_y_ref, R_ref, I_ref, ...
    left_boundary_buffer, image_generation_parameters.camera_design.x_pixel_number - right_boundary_buffer, ...
    bottom_boundary_buffer, image_generation_parameters.camera_design.y_pixel_number - top_boundary_buffer);

% --------------------------
% display results
% --------------------------
figure, 
imagesc(im_ref), colormap(flipud(gray))
hold on
plot(pos_ref_dots.x, pos_ref_dots.y, '*')
plot(X_ref, Y_ref, 'o')
annotate_image(gcf, gca);
xlim([left_boundary_buffer, image_generation_parameters.camera_design.x_pixel_number - right_boundary_buffer])
ylim([bottom_boundary_buffer, image_generation_parameters.camera_design.y_pixel_number - top_boundary_buffer])
set(gca, 'ydir', 'normal')
title('Reference')

%%
% --------------------------
% Gradient Image
% --------------------------
fprintf('Gradient Images\n');

% load image
fid = fopen(fullfile(current_image_directory, 'bos_pattern_image_2.bin'));
a = fread(fid, 'single');
fclose(fid);
im_grad = reshape(a, image_generation_parameters.camera_design.x_pixel_number, image_generation_parameters.camera_design.y_pixel_number);
im_grad = im_grad';
% im_grad = imread(fullfile(current_image_directory, 'bos_pattern_image_2.tif'));
% % flip image upside down so bottom pixel is y = 1
% im_grad = flipud(im_grad);
% flip image left to right
im_grad = fliplr(im_grad);
% mask image if required
if image_masking
    im_grad = double(im_grad) .* mask;
end
% subtract background if required
if background_subtraction
    im_grad = im_grad - im_bg;
end

% set image pixels outside buffer to zero
im_grad(:, 1:left_boundary_buffer) = 0;
im_grad(:, image_generation_parameters.camera_design.x_pixel_number - right_boundary_buffer:end) = 0;
im_grad(1:bottom_boundary_buffer, :) = 0;
im_grad(image_generation_parameters.camera_design.y_pixel_number - top_boundary_buffer:end, :) = 0;

% identify dots
[SIZE_grad.XYDiameter, SIZE_grad.peaks, SIZE_grad.mapsizeinfo, SIZE_grad.locxy, SIZE_grad.mapint] = combined_ID_size_apriori_10(im_grad, pos_ref_dots.x, pos_ref_dots.y, dot_diameter+2, subpixel_fit, default_iwc, min_area, W_area, W_intensity, W_distance);

% extract dot properties from sizing results
[X_grad, Y_grad, Z_grad, d_x_grad, d_y_grad, R_grad, I_grad] = extract_dot_properties_02(SIZE_grad.XYDiameter);

% remove dots outside ROI
[X_grad, Y_grad, Z_grad, d_x_grad, d_y_grad, R_grad, I_grad] = nan_dots_outside_ROI_02(X_grad, Y_grad, Z_grad, d_x_grad, d_y_grad, R_grad, I_grad, ...
    left_boundary_buffer, image_generation_parameters.camera_design.x_pixel_number - right_boundary_buffer, ...
    bottom_boundary_buffer, image_generation_parameters.camera_design.y_pixel_number - top_boundary_buffer);

% --------------------------
% display results
% --------------------------
figure, 
imagesc(im_grad), colormap(flipud(gray))
hold on
plot(pos_ref_dots.x, pos_ref_dots.y, '*')
plot(X_grad, Y_grad, 'o')
annotate_image(gcf, gca);
xlim([left_boundary_buffer, image_generation_parameters.camera_design.x_pixel_number - right_boundary_buffer])
ylim([bottom_boundary_buffer, image_generation_parameters.camera_design.y_pixel_number - top_boundary_buffer])
set(gca, 'ydir', 'normal')
title('Gradient')

%% track dots to estimate displacements
s_radius = 3;
fprintf('tracking dot locations between reference image and first grad image\n');
% run tracking
[tracks_dots]=weighted_nearest_neighbor3D(X_grad, X_ref, X_grad, Y_grad, Y_ref, Y_grad,...
    Z_grad, Z_ref, Z_grad, d_x_grad, d_x_ref, I_grad, I_ref, weights, s_radius);

x_track_dots = tracks_dots(:,1);
y_track_dots = tracks_dots(:,3);
u_track_dots = tracks_dots(:,2) - tracks_dots(:,1);
v_track_dots = tracks_dots(:,4) - tracks_dots(:,3);
% remove boundary tracks
[x_track_dots, y_track_dots, u_track_dots, v_track_dots] = remove_boundary_points_scattered(x_track_dots, y_track_dots, u_track_dots, v_track_dots, image_generation_parameters.camera_design.x_pixel_number, image_generation_parameters.camera_design.y_pixel_number, left_boundary_buffer, right_boundary_buffer, bottom_boundary_buffer, top_boundary_buffer);

% plot tracks
figure
quiver(x_track_dots, y_track_dots, u_track_dots/scale_factor, v_track_dots/scale_factor, 'AutoScale', 'off') 
annotate_image(gcf, gca);
title('Tracked Dots')
if save_figures
    saveas(gcf, fullfile(figure_save_directory, 'track-dots.png'), 'png');
    saveas(gcf, fullfile(figure_save_directory, 'track-dots.eps'), 'epsc');    
end

%% calculate wedge flow field

% name of file containing density data
file_name = '/scratch/shannon/c/aether/Projects/BOS/crlb/analysis/data/density-gradient-files/wedge_nx=234_ny=84_nz=51.mat';
% load data
wedge_data = load(file_name);
% extract 2D slice of density
rho = wedge_data.rho_3d(:,:,1);
% flip density field
% rho = fliplr(flipud(rho));
rho = rot90(rho, 2);

% create X,Y co-ordinate grid of density values
x_rho = image_generation_parameters.density_gradients.origin(1) + (1:image_generation_parameters.density_gradients.num_grid_points(1)) * image_generation_parameters.density_gradients.spacing(1);
y_rho = image_generation_parameters.density_gradients.origin(2) + (1:image_generation_parameters.density_gradients.num_grid_points(2)) * image_generation_parameters.density_gradients.spacing(2);
z_rho = image_generation_parameters.density_gradients.origin(3) + image_generation_parameters.density_gradients.volume_dimensions(3)/2;

% calculate magnification of the density gradient field on the image plane
M_rho = image_generation_parameters.lens_design.magnification * image_generation_parameters.lens_design.object_distance/z_rho;

% calculate co-ordinates of the density gradient field on the image plane
% (pix.)
[X_rho, Y_rho] = meshgrid(x_rho, y_rho);
X_rho = X_rho * M_rho * 1/image_generation_parameters.camera_design.pixel_pitch; 
Y_rho = Y_rho * M_rho * 1/image_generation_parameters.camera_design.pixel_pitch; 

% ensure that co-ordinates are positive integers
X_rho = floor(X_rho + image_generation_parameters.camera_design.x_pixel_number/2);
Y_rho = floor(Y_rho + image_generation_parameters.camera_design.y_pixel_number/2);

% calculate density gradients along x and y
rho_x = socdiff(rho, image_generation_parameters.density_gradients.spacing(1)*1e-6, 2);
rho_y = socdiff(rho, image_generation_parameters.density_gradients.spacing(2)*1e-6, 1);

% calculate second gradients of density
rho_xx = socdiff(rho_x, image_generation_parameters.density_gradients.spacing(1)*1e-6, 2);
rho_xy = socdiff(rho_x, image_generation_parameters.density_gradients.spacing(2)*1e-6, 1);
rho_yy = socdiff(rho_y, image_generation_parameters.density_gradients.spacing(1)*1e-6, 1);

% calculate factor by which the midpoint of the density gradient is
% magnified with respect to the dot pattern
magnification_factor = image_generation_parameters.lens_design.object_distance/z_rho;

% interpolate density gradients onto the tracks
rho_x_interp = interp2(X_rho, Y_rho, rho_x, x_track_dots, y_track_dots);
rho_y_interp = interp2(X_rho, Y_rho, rho_y, x_track_dots, y_track_dots);

% calculate second gradients density at the interrogation points using interpolation
rho_xx_interp = interp2(X_rho, Y_rho, rho_xx, x_track_dots, y_track_dots);
rho_yy_interp = interp2(X_rho, Y_rho, rho_yy, x_track_dots, y_track_dots);
rho_xy_interp = interp2(X_rho, Y_rho, rho_xy, x_track_dots, y_track_dots);

% gladstone dale constant (m^3/kg)
gladstone_dale = 0.225e-3;

% calculate ambient refractive index
n_0 = 1 + gladstone_dale * double(rho(end));

%% check if theoretical density gradients are aligned with the tracks
figure
quiver((x_track_dots - x_pixel_number/2) * magnification_factor + x_pixel_number/2, (y_track_dots - y_pixel_number/2) * magnification_factor + y_pixel_number/2, ...
    u_track_dots * magnification_factor/scale_factor, v_track_dots * magnification_factor/scale_factor, 'AutoScale', 'off')
hold on
quiver(X_rho, Y_rho, rho_x/0.1e2, rho_y/0.1e2, 'AutoScale', 'off')
annotate_image(gcf, gca);
legend('Tracks', 'Density Gradients', 'Location', 'northoutside', 'Orientation', 'horizontal')

%% calculate position estimation variance for tracked dots

fprintf('Calculating position estimation variance with a noise level of %.2f \n', noise_gray_level*100);

num_tracks = size(tracks_dots, 1);

xc_tracked_ref = ones(num_tracks,1) * NaN;
yc_tracked_ref = ones(num_tracks,1) * NaN;
xc_tracked_grad = ones(num_tracks,1) * NaN;
yc_tracked_grad = ones(num_tracks,1) * NaN;

d_x_tracked_ref = ones(num_tracks,1) * NaN;
d_y_tracked_ref = ones(num_tracks,1) * NaN;
d_x_tracked_grad = ones(num_tracks,1) * NaN;
d_y_tracked_grad = ones(num_tracks,1) * NaN;

R_tracked_ref = ones(num_tracks,1) * NaN;
R_tracked_grad = ones(num_tracks,1) * NaN;

pos_std_x_ref = ones(num_tracks,1) * NaN;
pos_std_y_ref = ones(num_tracks,1) * NaN;
pos_std_x_grad = ones(num_tracks,1) * NaN;
pos_std_y_grad = ones(num_tracks,1) * NaN;

pos_std_x_ref_theory = ones(num_tracks,1) * NaN;
pos_std_y_ref_theory = ones(num_tracks,1) * NaN;
pos_std_x_grad_theory = ones(num_tracks,1) * NaN;
pos_std_y_grad_theory = ones(num_tracks,1) * NaN;

pos_std_x_grad_hybrid = ones(num_tracks,1) * NaN;
pos_std_y_grad_hybrid = ones(num_tracks,1) * NaN;

pos_std_x_ref_measured = ones(num_tracks,1) * NaN;
pos_std_y_ref_measured = ones(num_tracks,1) * NaN;
pos_std_x_grad_measured = ones(num_tracks,1) * NaN;
pos_std_y_grad_measured = ones(num_tracks,1) * NaN;

amplification_ratio_x = ones(num_tracks,1) * NaN;
amplification_ratio_y = ones(num_tracks,1) * NaN;
amplification_ratio_x_theory = ones(num_tracks,1) * NaN;
amplification_ratio_y_theory = ones(num_tracks,1) * NaN;

%%

% loop through all tracked dots
for track_index = 1:num_tracks
    %% display progress to user
    if rem(track_index, 1000) == 0
        fprintf('Track number: %d\n', track_index);
    end

    %% extract dot index
    
    % reference 
    dot_index_ref = tracks_dots(track_index, 12);
    % gradient 
    dot_index_grad = tracks_dots(track_index, 11);
    
    %% extract intensity profile
    
    % reference 
    im_crop_ref = SIZE_ref.mapint{dot_index_ref};
    % gradient 
    im_crop_grad = SIZE_grad.mapint{dot_index_grad};    

    %% extract dot centroid
    
    % reference
    xc_tracked_ref(track_index) = X_ref(dot_index_ref);
    yc_tracked_ref(track_index) = Y_ref(dot_index_ref);
    xc_crop_ref = xc_tracked_ref(track_index) - SIZE_ref.locxy(dot_index_ref,1) + 1;
    yc_crop_ref = yc_tracked_ref(track_index) - SIZE_ref.locxy(dot_index_ref,2) + 1;

    % gradient
    xc_tracked_grad(track_index) = X_grad(dot_index_grad);
    yc_tracked_grad(track_index) = Y_grad(dot_index_grad);
    xc_crop_grad = xc_tracked_grad(track_index) - SIZE_grad.locxy(dot_index_grad,1) + 1;
    yc_crop_grad = yc_tracked_grad(track_index) - SIZE_grad.locxy(dot_index_grad,2) + 1;
    
    %% extract dot diameter
    
    % reference
    d_x_tracked_ref(track_index) = d_x_ref(dot_index_ref);
    d_y_tracked_ref(track_index) = d_y_ref(dot_index_ref);
    
    % gradient
    d_x_tracked_grad(track_index) = d_x_grad(dot_index_grad);
    d_y_tracked_grad(track_index) = d_y_grad(dot_index_grad);
    
    %% extract correlation coefficient
    
    R_tracked_ref(track_index) = R_ref(dot_index_ref);
    R_tracked_grad(track_index) = R_grad(dot_index_grad);

    %% calculate position estimation variance from monte-carlo
    
    % reference
    [err_var_x, err_var_y, ~, ~, ~] = ...
        calculate_position_estimation_variance_monte_carlo(im_crop_ref, xc_crop_ref, yc_crop_ref, noise_gray_level, ...
                                                           num_trials, subpixel_fit, plot_error_histogram);
	pos_std_x_ref(track_index) = sqrt(err_var_x);
    pos_std_y_ref(track_index) = sqrt(err_var_y);
    
    % gradient
    [err_var_x, err_var_y, ~, ~, ~] = ...
        calculate_position_estimation_variance_monte_carlo(im_crop_grad, xc_crop_grad, yc_crop_grad, noise_gray_level, ...
                                                           num_trials, subpixel_fit, plot_error_histogram);
    
    pos_std_x_grad(track_index) = sqrt(err_var_x);
    pos_std_y_grad(track_index) = sqrt(err_var_y);

    %% calculate amplification ratio
    
    % x
    amplification_ratio_x(track_index) = pos_std_x_grad(track_index)/pos_std_x_ref(track_index);
    % y
    amplification_ratio_y(track_index) = pos_std_y_grad(track_index)/pos_std_y_ref(track_index);
    
    %% calculate crlb from theory
    
    % reference
    [crlb_x, crlb_y] = calculate_crlb_bos_05(im_crop_ref, n_0, 0, 0, ...
        0, 0, 0, ...
        0.5 * (d_x_tracked_ref(track_index) + d_y_tracked_ref(track_index)), image_generation_parameters.lens_design.magnification, ...
        image_generation_parameters.lens_design.aperture_f_number, Z_D*1e-3, Z_g*1e-3, ...
        image_generation_parameters.camera_design.pixel_pitch*1e-6, noise_gray_level);
    
    pos_std_x_ref_theory(track_index) = sqrt(crlb_x);
    pos_std_y_ref_theory(track_index) = sqrt(crlb_y);
    
    % gradient
    [crlb_x, crlb_y] = calculate_crlb_bos_05(im_crop_ref, n_0, rho_x_interp(track_index), rho_y_interp(track_index), ...
        rho_xx_interp(track_index), rho_xy_interp(track_index), rho_yy_interp(track_index), ...
        0.5 * (d_x_tracked_ref(track_index) + d_y_tracked_ref(track_index)), image_generation_parameters.lens_design.magnification, ...
        image_generation_parameters.lens_design.aperture_f_number, Z_D*1e-3, Z_g*1e-3, ...
        image_generation_parameters.camera_design.pixel_pitch*1e-6, noise_gray_level);
    
    pos_std_x_grad_theory(track_index) = sqrt(crlb_x);
    pos_std_y_grad_theory(track_index) = sqrt(crlb_y);
    
    %% calculate amplification ratio from theory
    
    % x
    amplification_ratio_x_theory(track_index) = pos_std_x_grad_theory(track_index)/pos_std_x_ref_theory(track_index);
    % y
    amplification_ratio_y_theory(track_index) = pos_std_y_grad_theory(track_index)/pos_std_y_ref_theory(track_index);
    
    %% calculate crlb from measured diameter
    
    % reference
%     pos_std_x_ref_measured(track_index) = (2 * sqrt(2*pi) * noise_gray_level / sum(im_crop_ref(:)) * d_tracked_ref(track_index)^2/16); % ^ 2;
%     pos_std_y_ref_measured(track_index) = (2 * sqrt(2*pi) * noise_gray_level / sum(im_crop_ref(:)) * d_tracked_ref(track_index)^2/16); % ^ 2;    
    pos_std_x_ref_measured(track_index) = 2 * sqrt(2*pi) * noise_gray_level / sum(im_crop_ref(:)) * ... 
                                d_x_tracked_ref(track_index)^(3/2) * d_y_tracked_ref(track_index)^(1/2) * ...
                                (1 - R_tracked_ref(track_index)^2)^0.25;    
    pos_std_y_ref_measured(track_index) = 2 * sqrt(2*pi) * noise_gray_level / sum(im_crop_ref(:)) * ... 
                                d_x_tracked_ref(track_index)^(1/2) * d_y_tracked_ref(track_index)^(3/2) * ...
                                (1 - R_tracked_ref(track_index)^2)^0.25;    

    % gradient
%     pos_std_x_grad_measured(track_index) = (2 * sqrt(2*pi) * noise_gray_level / sum(im_crop_grad(:)) * d_tracked_grad(track_index)^2/16); % ^ 2;
%     pos_std_x_grad_measured(track_index) = (2 * sqrt(2*pi) * noise_gray_level / sum(im_crop_grad(:)) * d_tracked_grad(track_index)^2/16); % ^ 2;
    pos_std_x_grad_measured(track_index) = 2 * sqrt(2*pi) * noise_gray_level / sum(im_crop_grad(:)) * ... 
                                d_x_tracked_grad(track_index)^(3/2) * d_y_tracked_grad(track_index)^(1/2) * ...
                                (1 - R_tracked_grad(track_index)^2)^0.25;    
    pos_std_y_grad_measured(track_index) = 2 * sqrt(2*pi) * noise_gray_level / sum(im_crop_grad(:)) * ... 
                                d_x_tracked_grad(track_index)^(1/2) * d_y_tracked_grad(track_index)^(3/2) * ...
                                (1 - R_tracked_grad(track_index)^2)^0.25;    
    
    %% calculate crlb from theoretical ratio and measured std of reference
    
    % x
    pos_std_x_grad_hybrid(track_index) = amplification_ratio_x_theory(track_index) * pos_std_x_ref(track_index);
    % y
    pos_std_y_grad_hybrid(track_index) = amplification_ratio_y_theory(track_index) * pos_std_y_ref(track_index);
    
end

fprintf('...done\n');

%% interpolate results onto a grid

% amplfication ratio, simulation
[X_ref_grid, Y_ref_grid, amplification_ratio_x_grid, amplification_ratio_y_grid] = ...
    interpolate_tracks(xc_tracked_ref, yc_tracked_ref, ...
    amplification_ratio_x, amplification_ratio_y, dot_spacing);

% amplfication ratio, theory
[~, ~, amplification_ratio_x_theory_grid, amplification_ratio_y_theory_grid] = ...
    interpolate_tracks(xc_tracked_ref, yc_tracked_ref, ...
    amplification_ratio_x_theory, amplification_ratio_y_theory, dot_spacing);

% dot diameters, gradient
[~, ~, d_x_tracked_grad_grid, ~] = ...
    interpolate_tracks(xc_tracked_ref, yc_tracked_ref, ...
    d_x_tracked_grad, d_x_tracked_grad, dot_spacing);

% dot diameters, gradient
[~, ~, d_y_tracked_grad_grid, ~] = ...
    interpolate_tracks(xc_tracked_ref, yc_tracked_ref, ...
    d_x_tracked_grad, d_y_tracked_grad, dot_spacing);

%% calculate pdf of uncertainty

% bin levels
edges_std = linspace(0, 0.2, 100);
% simulation
[N_std_sim, ~] = histcounts([pos_std_x_grad; pos_std_y_grad], edges_std, 'Normalization', 'pdf');
% theory
[N_std_theory, ~] = histcounts([pos_std_x_grad_theory; pos_std_y_grad_theory], edges_std, 'Normalization', 'pdf');
% ratio from theory
[N_std_hybrid, ~] = histcounts([pos_std_x_grad_hybrid; pos_std_y_grad_hybrid], edges_std, 'Normalization', 'pdf');

%% calculate pdf of amplification ratio

% bin levels
edges_ratio = linspace(1, 1.5, 100);
% simulation
[N_ratio_sim, ~] = histcounts([amplification_ratio_x; amplification_ratio_y], edges_ratio, 'Normalization', 'pdf');
% theory
[N_ratio_theory, ~] = histcounts([amplification_ratio_x_theory; amplification_ratio_y_theory], edges_ratio, 'Normalization', 'pdf');

%% calculate rms of uncertainty

% simulation
rms_std_sim = rms([pos_std_x_grad; pos_std_y_grad], 'omitnan');
% theory
rms_std_theory = rms([pos_std_x_grad_theory; pos_std_y_grad_theory], 'omitnan');
% hybrid
rms_std_hybrid = rms([pos_std_x_grad_hybrid; pos_std_y_grad_hybrid], 'omitnan');

%% calculate rms of amplifcation ratio

% simulation
rms_ratio_sim = rms([amplification_ratio_x; amplification_ratio_y], 'omitnan');
% theory
rms_ratio_theory = rms([amplification_ratio_x_theory; amplification_ratio_y_theory], 'omitnan');

%% plot contour of measured diameters for gradient image

% display contours
cmin_current = dot_size_pixels;
cmax_current = dot_size_pixels + 2;
contour_levels = linspace(cmin_current, cmax_current, 100);

figure
contourf(X_ref_grid, Y_ref_grid, sqrt(d_x_tracked_grad_grid.^2 + d_y_tracked_grad_grid.^2), ...
    contour_levels, 'edgecolor', 'none')
colorcet('fire', 'N', 100, 'reverse', 1);
% caxis([cmin_current, cmax_current]);
caxis([4, 6]);
colorbar
annotate_image(gcf, gca);
% axis([xmin_mask xmax_mask ymin_mask ymax_mask])
box on
title('Dot Diameters')
set(gcf, 'Position', [360   584   441   316])
if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'diameter-contour', [1, 0, 1]);
end

%% plot contour of amplification ratio magnitude from simulations

% display contours
cmin_current = 1;
cmax_current = 2;
contour_levels = linspace(cmin_current, cmax_current, 100);

figure
contourf(X_ref_grid, Y_ref_grid, sqrt(amplification_ratio_x_grid.^2 + amplification_ratio_y_grid.^2)/sqrt(2), ...
    contour_levels, 'edgecolor', 'none')
colorcet('fire', 'N', 100, 'reverse', 1);
caxis([cmin_current, cmax_current]);
colorbar
annotate_image(gcf, gca);
% axis([xmin_mask xmax_mask ymin_mask ymax_mask])
box on
title('Ratio, Simulation')
set(gcf, 'Position', [360   584   441   316])
if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'amplification-ratio-contour', [1, 0, 1]);
end

%% plot contour of amplification ratio magnitude from theory

% display contours
cmin_current = 1;
cmax_current = 2;
contour_levels = linspace(cmin_current, cmax_current, 100);

figure
contourf(X_ref_grid, Y_ref_grid, sqrt(amplification_ratio_x_theory_grid.^2 + amplification_ratio_y_theory_grid.^2)/sqrt(2), ...
    contour_levels, 'edgecolor', 'none')
colorcet('fire', 'N', 100, 'reverse', 1);
caxis([cmin_current, cmax_current]);
colorbar
annotate_image(gcf, gca);
% axis([xmin_mask xmax_mask ymin_mask ymax_mask])
box on
title('Ratio, Theory')
set(gcf, 'Position', [360   584   441   316])
if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'amplification-ratio-theory-contour', [1, 0, 1]);
end

%% plot scatter plot of amplification ratio magnitude

minval = 1;
maxval = 1.5;

figure
% plot(sqrt(amplification_ratio_x_theory.^2 + amplification_ratio_y_theory.^2)/sqrt(2), ...
%     sqrt(amplification_ratio_x.^2 + amplification_ratio_y.^2)/sqrt(2), ...
%     'o', 'markersize', 3, 'markeredgecolor', colors(1,:), 'markerfacecolor', colors(1,:))
plot(sqrt(amplification_ratio_x_theory_grid.^2 + amplification_ratio_y_theory_grid.^2)/sqrt(2), ...
    sqrt(amplification_ratio_x_grid.^2 + amplification_ratio_y_grid.^2)/sqrt(2), ...
    'o', 'markersize', 3, 'markeredgecolor', colors(1,:), 'markerfacecolor', colors(1,:))

axis equal
hold on
plot([minval maxval], [minval maxval])
axis([minval maxval minval maxval])
grid on
xlabel('Theory (pix.)')
ylabel('Simulation (pix.)')
title('Amplification Ratio')
set(gcf, 'Position', [360 559 417 341]);

if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'amplification-ratio-scatter', [1, 1, 1]);
end

%% plot histograms of position uncertainty

x_max = 0.2;
y_max = 150;

figure
% simulation
plot(edges_std(1:end-1), N_std_sim, 'k');
hold on
plot(rms_std_sim * [1, 1], [0, y_max], 'k--');
% theory
plot(edges_std(1:end-1), N_std_theory, 'color', colors(1,:));
plot(rms_std_theory * [1, 1], [0, y_max], 'color', colors(1,:), 'linestyle', '--');
% hybrid
plot(edges_std(1:end-1), N_std_hybrid, 'color', colors(2,:));
plot(rms_std_hybrid * [1, 1], [0, y_max], 'color', colors(2,:), 'linestyle', '--');

xlim([0 x_max])
ylim([0 y_max])

lgd = legend('\sigma_{X_0}, Sim.', 'RMS \sigma_{X_0}, Sim.', '\sigma_{X_0}, Theory', 'RMS \sigma_{X_0}, Theory', ...
    '\sigma_{X_0}, Hybrid', 'RMS \sigma_{X_0}, Hybrid');
lgd.FontSize = 10;

xlabel('\sigma_{X_0} (pix.)')
set(gcf, 'Position', [360   584   441   316])

if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'uncertainty-hist-theory-sim-hybrid', [1 1 1]);
end

%% plot histograms of amplification ratio

x_max = 1.2;
y_max = 150;

figure
plot(edges_ratio(1:end-1), N_ratio_sim, 'k');
hold on
plot(rms_ratio_sim * [1, 1], [0, y_max], 'k--');
plot(edges_ratio(1:end-1), N_ratio_theory, 'color', colors(1,:));
plot(rms_ratio_theory * [1, 1], [0, y_max], 'color', colors(1,:), 'linestyle', '--');

xlim([1 x_max])
ylim([0 y_max])

legend('AR, Sim.', 'RMS AR, Sim.', 'AR, Theory.', 'RMS AR, Theory.')
xlabel('AR')
set(gcf, 'Position', [360   584   441   316])

if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'ratio-hist-theory-sim', [1 1 1]);
end

%%

% %% create scattered interpolants
% 
% % simulation, x
% F_x_sim = scatteredInterpolant(xc_tracked_ref, yc_tracked_ref, pos_std_x_grad);
% F_x_sim.Method = 'nearest';
% F_x_sim.ExtrapolationMethod = 'none';
% % simulation, y
% F_y_sim = scatteredInterpolant(xc_tracked_ref, yc_tracked_ref, pos_std_y_grad);
% F_y_sim.Method = 'nearest';
% F_y_sim.ExtrapolationMethod = 'none';
% 
% % theory, x
% F_x_theory = scatteredInterpolant(xc_tracked_ref, yc_tracked_ref, pos_std_x_grad_theory);
% F_x_theory.Method = 'nearest';
% F_x_theory.ExtrapolationMethod = 'none';
% 
% % theory, y
% F_y_theory = scatteredInterpolant(xc_tracked_ref, yc_tracked_ref, pos_std_y_grad_theory);
% F_y_theory.Method = 'nearest';
% F_y_theory.ExtrapolationMethod = 'none';
% 
% % theory with measured diameter, x
% F_x_measured = scatteredInterpolant(xc_tracked_ref, yc_tracked_ref, pos_std_x_grad_measured);
% F_x_measured.Method = 'nearest';
% F_x_measured.ExtrapolationMethod = 'none';
% 
% % theory with measured diameter, y
% F_y_measured = scatteredInterpolant(xc_tracked_ref, yc_tracked_ref, pos_std_y_grad_measured);
% F_y_measured.Method = 'nearest';
% F_y_measured.ExtrapolationMethod = 'none';
% 
% %% compare standard deviation on a slice
% 
% % co-ordinates on the slice
% x_slice = 200:dot_spacing:800;
% y_slice = 252 * ones(size(x_slice));
% 
% % plot results
% figure
% subplot(1,2,1)
% plot(x_slice, F_x_sim(x_slice, y_slice), 'o')
% hold on
% plot(x_slice, F_x_theory(x_slice, y_slice), 'o')
% plot(x_slice, F_x_measured(x_slice, y_slice), 'o')
% ylim([0 1])
% xlabel('X (pix.)')
% ylabel('\sigma_{X_0} (pix.)')
% legend('Simulation', 'CRLB', 'CRLB - Measured')
% legend boxon
% title(['Y = ' num2str(y_slice(1)) ' pix.'])
% 
% subplot(1,2,2)
% plot(x_slice, F_y_sim(x_slice, y_slice), 'o')
% hold on
% plot(x_slice, F_y_theory(x_slice, y_slice), 'o')
% ylim([0 1])
% xlabel('X (pix.)')
% ylabel('\sigma_{Y_0} (pix.)')
% legend('Simulation', 'CRLB')
% title(['Y = ' num2str(y_slice(1)) ' pix.'])
% 
% set(gcf, 'Position', [64         433        1109         403])
% 
% if save_figures
%     saveas(gcf, fullfile(figure_save_directory, ['pos-std-comparison-slice-y=' num2str(y_slice(1)) '.png']), 'png');
%     saveas(gcf, fullfile(figure_save_directory, ['pos-std-comparison-slice-y=' num2str(y_slice(1)) '.eps']), 'epsc');
% end
%%
% %% plot amplification ratio magnitude
% 
% % calculate amplification ratios
% AR_x = (crlb_ratio_measured_I .* sqrt(crlb_x_ratio_measured./crlb_y_ratio_measured))';
% AR_y = (crlb_ratio_measured_I .* sqrt(crlb_y_ratio_measured./crlb_x_ratio_measured))';
% 
% % interpolate results onto a grid
% [X_ref_grid, Y_ref_grid, AR_x_grid, AR_y_grid] = interpolate_tracks(X_all_ref(:,1), Y_all_ref(:,1), AR_x, AR_y, dot_spacing);
% 
% % display contours
% cmin_current = 1.5;
% cmax_current = 1.9;
% contour_levels = linspace(cmin_current, cmax_current, 100);
% figure
% contourf(X_ref_grid, Y_ref_grid, sqrt(AR_x_grid.^2 + AR_y_grid.^2), contour_levels, 'edgecolor', 'none')
% colorcet('fire', 'N', 100, 'reverse', 1);
% caxis([cmin_current, cmax_current]);
% colorbar
% annotate_image(gcf, gca);
% axis([xmin_mask xmax_mask ymin_mask ymax_mask])
% box on
% title('Ratio')
% set(gcf, 'Position', [360   584   441   316])
% 
% %% plot distribution of mean measured dot diameter on the image
% 
% % display image
% figure
% imagesc(im_ref)
% colormap(flipud(gray))
% annotate_image(gcf, gca);
% % axis([xmin_mask xmax_mask ymin_mask ymax_mask])
% hold on
% 
% plot_colored_circles(gcf, X_grad, Y_grad, d_grad, 0.75*dot_diameter, 1.5*dot_diameter, dot_diameter/2);
% set(gca, 'ydir', 'normal')
% title({'Dot Diameter - Experiment'; ['Min: ' num2str(0.75*dot_diameter, '%.2f') ' pix., Max: ' num2str(1.25*dot_diameter, '%.2f') ' pix.']})
% set(gcf, 'Position', [360   275   741   625])
% set(gcf, 'Resize', 'off')
% if save_figures
%     saveas(gcf, fullfile(figure_save_directory, 'dot-diameter-exp-spatial.png'), 'png');
%     saveas(gcf, fullfile(figure_save_directory, 'dot-diameter-exp-spatial.eps'), 'epsc');
% end
% 
% %% plot distribution of position estimation variance on the image
% 
% % ------------------------------------
% % plot circles with global color bar
% % ------------------------------------
% 
% % display image
% figure
% imagesc(im_grad)
% colormap(flipud(gray))
% annotate_image(gcf, gca);
% % axis([xmin_mask xmax_mask ymin_mask ymax_mask])
% hold on
% 
% plot_colored_circles(gcf, X_all, Y_all, sqrt(err_var_x_all + err_var_y_all), global_cmin, global_cmax, dot_diameter/2);
% % plot_colored_circles(gcf, X_grad, Y_grad, sqrt(err_var_x_all + err_var_y_all), cmin, cmax, dot_diameter/2);
% ylim([100 400])
% set(gca, 'ydir', 'normal')
% title({'\sigma_{X_0} - Simulation'; ['Min: ' num2str(global_cmin, '%.2f') ' pix., Max: ', num2str(global_cmax, '%.2f') ' pix.']})
% set(gcf, 'Position', [360   275   741   625])
% set(gcf, 'Resize', 'off')
% if save_figures
%     saveas(gcf, fullfile(figure_save_directory, 'pos-std-exp-spatial-global.png'), 'png');
%     saveas(gcf, fullfile(figure_save_directory, 'pos-std-exp-spatial-global.eps'), 'epsc');
% end
% 
% % return;
% % ------------------------------------
% % plot circles with local color bar
% % ------------------------------------
% 
% local_cmin = prctile(sqrt(err_var_x_all + err_var_y_all), 0.1);
% local_cmax = prctile(sqrt(err_var_x_all + err_var_y_all), 0.9);
% 
% % display image
% figure
% imagesc(im_grad)
% colormap(flipud(gray))
% annotate_image(gcf, gca);
% % axis([xmin_mask xmax_mask ymin_mask ymax_mask])
% hold on
% 
% plot_colored_circles(gcf, X_all, Y_all, sqrt(err_var_x_all + err_var_y_all), local_cmin, local_cmax, dot_diameter/2);
% % plot_colored_circles(gcf, X_grad, Y_grad, sqrt(err_var_x_all + err_var_y_all), cmin, cmax, dot_diameter/2);
% ylim([100 400])
% set(gca, 'ydir', 'normal')
% title({'\sigma_{X_0} - Simulation'; ['Min: ' num2str(local_cmin, '%.2f') ' pix., Max: ', num2str(local_cmax, '%.2f') ' pix.']})
% set(gcf, 'Position', [360   275   741   625])
% set(gcf, 'Resize', 'off')
% if save_figures
%     saveas(gcf, fullfile(figure_save_directory, 'pos-std-exp-spatial-local.png'), 'png');
%     saveas(gcf, fullfile(figure_save_directory, 'pos-std-exp-spatial-local.eps'), 'epsc');
% end
% 
% %% calculate crlb
% fprintf('calculating crlb \n');
% 
% % number of dots for which the crlb needs to be calculated
% num_dots = size(X_ref,1);
% 
% % declare arrays to hold results
% crlb_x_all = ones(1, num_dots) * NaN;
% crlb_x_all_measured = ones(1, num_dots) * NaN;
% crlb_y_all = ones(1, num_dots) * NaN;
% rho_x_interp = ones(1, num_dots) * NaN;
% rho_y_interp = ones(1, num_dots) * NaN;
% rho_xx_interp = ones(1, num_dots) * NaN;
% rho_xy_interp = ones(1, num_dots) * NaN;
% rho_yy_interp = ones(1, num_dots) * NaN;
% 
% %% loop through all dots and estimate crlb
% 
% for track_index = 1:num_tracks
%     %% display progress to user
%     if rem(track_index, 1000) == 0
%         fprintf('Track number: %d\n', track_index);
%     end
% 
%     %% extract dot index
%     
%     % reference 
%     dot_index_ref = tracks_dots(track_index, 11);
%     % gradient 
%     dot_index_grad = tracks_dots(track_index, 12);
%     
%     %% extract intensity profile
%     
%     % reference 
%     im_crop_ref = SIZE_ref.mapint{dot_index_ref};
%     % gradient 
%     im_crop_grad = SIZE_grad.mapint{dot_index_grad};
%     
%     %% extract dot centroid
%     
%     % reference
%     xc_tracked_ref(track_index) = X_ref(dot_index_ref);
%     yc_tracked_ref(track_index) = Y_ref(dot_index_ref);
%     xc_crop_ref = xc_tracked_ref(track_index) - SIZE_ref.locxy(dot_index_ref,1) + 1;
%     yc_crop_ref = yc_tracked_ref(track_index) - SIZE_ref.locxy(dot_index_ref,2) + 1;
% 
%     % gradient
%     xc_tracked_grad(track_index) = X_grad(dot_index_grad);
%     yc_tracked_grad(track_index) = Y_grad(dot_index_grad);
%     xc_crop_grad = xc_tracked_grad(track_index) - SIZE_grad.locxy(dot_index_grad,1) + 1;
%     yc_crop_grad = yc_tracked_grad(track_index) - SIZE_grad.locxy(dot_index_grad,2) + 1;
%     
% end
% 
% for p_ref = 1:num_dots %size(X_all, 1)
%     % display progress to user
%     if rem(p_ref, 1000) == 0
%         fprintf('Dot number: %d\n', p_ref);
%     end
%     
%     track_index = find(tracks_dots(:,12) == p_ref);
%     if isempty(track_index)
%         continue;
%     end
%     p_grad = tracks_dots(track_index, 11);
%     % extract the intensity map for this dot
% %     im_crop = SIZE_ref.mapint{p_ref};
%     im_crop = SIZE_grad.mapint{p_grad};
%     
%     % if the intensity map is blank, then continue to the next dot
%     if isempty(im_crop) || sum(im_crop(:)) == 0
%         continue;
%     end
%     
% %     % extract peak intensity if it is finite
% %     if isfinite(I_ref(p_ref))
% %         I_max = I_ref(p_ref);
% %     else
% %         I_max = max(im_crop(:));
% %     end
% %     I_max = max(im_crop(:));
%     
%     % calculate first gradients of density at the interrogation points using interpolation
%     rho_x_interp(p_ref) = interp2(X_rho, Y_rho, rho_x, X_ref(p_ref), Y_ref(p_ref));
%     rho_y_interp(p_ref) = interp2(X_rho, Y_rho, rho_y, X_ref(p_ref), Y_ref(p_ref));
%     
%     % calculate second gradients density at the interrogation points using interpolation
%     rho_xx_interp(p_ref) = interp2(X_rho, Y_rho, rho_xx, X_ref(p_ref), Y_ref(p_ref));
%     rho_yy_interp(p_ref) = interp2(X_rho, Y_rho, rho_yy, X_ref(p_ref), Y_ref(p_ref));
%     rho_xy_interp(p_ref) = interp2(X_rho, Y_rho, rho_xy, X_ref(p_ref), Y_ref(p_ref));
%      
%     % calculate crlb
% %     [crlb_x_all(p_ref), crlb_y_all(p_ref)] = calculate_crlb_bos(im_crop, rho_xx_interp(p_ref), rho_xy_interp(p_ref), rho_yy_interp(p_ref), d_ref(p_ref), image_generation_parameters.lens_design.magnification, image_generation_parameters.lens_design.aperture_f_number, Z_D*1e-3, Z_g*1e-3, image_generation_parameters.camera_design.pixel_pitch*1e-6, noise_level);
% %     [crlb_x_all(p_ref), crlb_y_all(p_ref)] = calculate_crlb_bos(im_crop, rho_xx_interp(p_ref), rho_xy_interp(p_ref), rho_yy_interp(p_ref), d_ref(p_ref), image_generation_parameters.lens_design.magnification, image_generation_parameters.lens_design.aperture_f_number, Z_D*1e-3, Z_g*1e-3, image_generation_parameters.camera_design.pixel_pitch*1e-6, noise_gray_level);
%     [crlb_x_all(p_ref), crlb_y_all(p_ref)] = calculate_crlb_bos_02(im_crop, rho_x_interp(p_ref), rho_y_interp(p_ref), rho_xx_interp(p_ref), rho_xy_interp(p_ref), rho_yy_interp(p_ref), d_ref(p_ref), image_generation_parameters.lens_design.magnification, image_generation_parameters.lens_design.aperture_f_number, Z_D*1e-3, Z_g*1e-3, image_generation_parameters.camera_design.pixel_pitch*1e-6, noise_gray_level);
%     crlb_x_all_measured(p_ref) = (2 * sqrt(2*pi) * noise_gray_level / sum(im_crop(:)) * d_grad(p_grad)^2/16) ^ 2;
% 
% end
% 
% %% plot distribution of crlb on the image
% 
% % ------------------------------------
% % plot circles with global color bar
% % ------------------------------------
% 
% % display the reference image
% figure
% imagesc(im_ref)
% colormap(flipud(gray))
% annotate_image(gcf, gca);
% hold on
% 
% % plot position standard deviation estimated from crlb
% plot_colored_circles(gcf, X_ref(:,1), Y_ref(:,1), sqrt(crlb_x_all + crlb_y_all), global_cmin, global_cmax, dot_diameter/2);
% 
% ylim([100 400])
% set(gca, 'ydir', 'normal')
% set(gcf, 'Position', [360   275   741   625])
% set(gcf, 'Resize', 'off')
% title({'\sigma_{X_0} - Theory'; ['Min: ' num2str(global_cmin, '%.2f') ' pix., Max: ', num2str(global_cmax, '%.2f') ' pix.']})
% if save_figures
%     saveas(gcf, fullfile(figure_save_directory, 'pos-std-theory-spatial-global.png'), 'png');
%     saveas(gcf, fullfile(figure_save_directory, 'pos-std-theory-spatial-global.eps'), 'epsc');
% end
% 
% % ------------------------------------
% % plot circles with local color bar
% % ------------------------------------
% 
% % calculate limits of the local color bar
% local_cmin = prctile(sqrt(crlb_x_all + crlb_y_all), 0.1);
% local_cmax = prctile(sqrt(crlb_x_all + crlb_y_all), 0.9);
% 
% % display the reference image
% figure
% imagesc(im_ref)
% colormap(flipud(gray))
% annotate_image(gcf, gca);
% hold on
% 
% % plot position standard deviation estimated from crlb
% plot_colored_circles(gcf, X_ref(:,1), Y_ref(:,1), sqrt(crlb_x_all + crlb_y_all), local_cmin, local_cmax, dot_diameter/2);
% 
% ylim([100 400])
% set(gca, 'ydir', 'normal')
% set(gcf, 'Position', [360   275   741   625])
% set(gcf, 'Resize', 'off')
% title({'\sigma_{X_0} - Theory'; ['Min: ' num2str(local_cmin, '%.2f') ' pix., Max: ', num2str(local_cmax, '%.2f') ' pix.']})
% if save_figures
%     saveas(gcf, fullfile(figure_save_directory, 'pos-std-theory-spatial-local.png'), 'png');
%     saveas(gcf, fullfile(figure_save_directory, 'pos-std-theory-spatial-local.eps'), 'epsc');
% end
% 
% %% match position estimation variance and crlb estimates for corresponding dots
% 
% % declare arrays to hold the crlb for the matched dots
% crlb_x_matched = ones(size(tracks_dots, 1), 1) * NaN;
% crlb_x_measured_matched = ones(size(tracks_dots, 1), 1) * NaN;
% crlb_y_matched = ones(size(tracks_dots, 1), 1) * NaN;
% 
% % declare arrays to hold the position estimation variance for the matched
% % dots
% pos_std_x_matched = ones(size(tracks_dots, 1), 1) * NaN;
% pos_std_y_matched = ones(size(tracks_dots, 1), 1) * NaN;
% 
% % loop through all dots in the tracks and find corresponding reference and
% % gradient dots
% for dot_index = 1:size(tracks_dots, 1)
%     % dot id in gradient image
%     p_grad = tracks_dots(dot_index,11);
%     % dot id in reference image
%     p_ref = tracks_dots(dot_index,12);
%     
% %     if abs(tracks_dots(dot_index, 2) - 596.7) < 1
% % %         flag = flag + 1
% %         continue;
% %     end
%     if isnan(p_ref) || isnan(p_grad)
%         continue;
%     end
%     
%     % crlb for the matched dot
%     crlb_x_matched(p_grad) = crlb_x_all(p_ref);
%     crlb_x_measured_matched(p_grad) = crlb_x_all_measured(p_ref);
%     crlb_y_matched(p_grad) = crlb_y_all(p_ref);
%     % position estimation variance for the matched dot
%     pos_std_x_matched(p_grad) = pos_std_x(p_grad);
% end
% 
% %% plot standard deviation of experiment vs theoretial predictions
% 
% colors = lines(2);
% figure
% plot(sqrt(crlb_x_matched), pos_std_x_matched, 'o', 'markersize', 3, 'markeredgecolor', colors(1,:)); %, 'markerfacecolor', colors(1,:))
% hold on
% plot(sqrt(crlb_x_measured_matched), pos_std_x_matched, 'o', 'markersize', 3, 'markeredgecolor', colors(2,:)); %, 'markerfacecolor', colors(1,:))
% axis equal
% hold on
% plot([0 1], [0 1])
% % axis([0.02 0.05 0.02 0.05])
% axis([0 1 0 1])
% grid on
% xlabel('\sigma_{X_0}, Theory (pix.)')
% ylabel('\sigma_{X_0}, Simulation (pix.)')
% legend('Theoretical Diameter', 'Measured Diameter', 'location', 'northoutside', 'Orientation', 'horizontal')
% % set(gcf, 'Position', [360 559 417 341]);
% set(gcf, 'Position', [360   388   618   512])
% % return;
% if save_figures
%     saveas(gcf, fullfile(figure_save_directory, 'scatter.png'), 'png');
%     saveas(gcf, fullfile(figure_save_directory, 'scatter.eps'), 'epsc');
% end
% 


%% save workspace to file
workspace_save_directory = fullfile(current_results_directory, 'workspace');
if ~exist(workspace_save_directory, 'dir')
    mkdir(workspace_save_directory);
end
 
% this is the name of the current script
script_name_full = mfilename('fullpath');
[pathstr, script_name, ext] = fileparts(script_name_full);

%%
% Get a list of all variables
allvars = whos;

% Identify the variables that ARE NOT graphics handles. This uses a regular
% expression on the class of each variable to check if it's a graphics object
tosave = cellfun(@isempty, regexp({allvars.class}, '^matlab\.(ui|graphics)\.'));

% Pass these variable names to save
% save('output.mat', allvars(tosave).name)
save(fullfile(workspace_save_directory, [script_name '.mat']), allvars(tosave).name);