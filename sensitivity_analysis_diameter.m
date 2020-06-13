clear
close all
clc

restoredefaultpath;
addpath('/scratch/shannon/c/aether/Projects/BOS/general-codes/matlab-codes/');
setup_default_settings;

%%

results_save_directory = fullfile('/scratch/shannon/c/aether/Projects/BOS/crlb/analysis/results/model/'); 
figure_save_directory = fullfile(results_save_directory, 'figures');
mkdir_c(figure_save_directory);

save_figures = true;

%%

% number of trials
num_trials = 1e6;

% number of bins for the histogram
num_bins = round(sqrt(num_trials));

%% expected value for parameters

% constants
K = 0.225e-3;
no = 1;
dr = 10e-6;

% variables
etae = 3/4;
Me = 1;
zde = 0.1;
fne = 16;
delze = 0.1;
del2rhoe = 2500;

%% range for each parameter

deta = 2*2/4;
dM = 2*0.5;
dzd = 2*0.05;
dfn = 2*8;
ddelz = 2*0.05;
ddel2rho = 2*1000;

%% initialize arrays for sensitivity coefficients

Aeta = nans(1, num_trials);
AM = nans(1, num_trials);
Azd = nans(1, num_trials);
Afn = nans(1, num_trials);
Adelz = nans(1, num_trials);
Adel2rho = nans(1, num_trials);

%%

parfor trial_index = 1:num_trials    
    % generate random numbers
    rand_nums = rand(1, 6) - 0.5;
    
    %% calculate parameter values
    
    eta = etae + rand_nums(1) * deta;
    M = Me + rand_nums(2) * dM;
    zd = zde + rand_nums(3) * dzd;
    fn = fne + rand_nums(4) * dfn;
    delz = delze + rand_nums(5) * ddelz;
    del2rho = del2rhoe + rand_nums(6) * ddel2rho;
    
    %% calculate sensitivity coefficient 
    
    % eta
    Aeta(trial_index) = abs(eta / sqrt(eta^2 + (del2rho^2 * delz^2 * K^2 * M^4 * zd^4) / (12 * dr^2 * fn^2 * (1 + M)^2 * no^2)));
  
    % M
    AM(trial_index) = abs((-((del2rho^2 * delz^2 * K^2 *M^4 *zd^4) / (6 * dr^2 * fn^2 * (1 + M)^3 * no^2)) + ...
        (del2rho^2 * delz^2 * K^2 * M^3 * zd^4)/(3 * dr^2 * fn^2 * (1 + M)^2 * no^2)) ...
        /(2 * sqrt(eta^2 + (del2rho^2 * delz^2 * K^2 * M^4 * zd^4)/(12 * dr^2 * fn^2 * (1 + M)^2 * no^2))));
  
    % zd
    Azd(trial_index) = abs((del2rho^2 * delz^2 * K^2 * M^4 * zd^3)/(6 * dr^2 * fn^2 * (1 + M)^2 * no^2 * ...
                       sqrt(eta^2 + (del2rho^2 * delz^2 * K^2 * M^4 * zd^4)/(12 * dr^2 * fn^2 * (1 + M)^2 * no^2))));

    % fn
    Afn(trial_index) = abs(-((del2rho^2 * delz^2 * K^2 * M^4 * zd^4)/(12 * dr^2 * fn^3 * (1 + M)^2 * no^2 * ...
                        sqrt(eta^2 + (del2rho^2 * delz^2 * K^2 *M^4 *zd^4) / (12 * dr^2 * fn^2 * (1 + M)^2 * no^2)))));    
    
    % delz
    Adelz(trial_index) = abs((del2rho^2 * delz * K^2 * M^4 * zd^4)/(12 * dr^2 * fn^2 * (1 + M)^2 * no^2 * ...
                         sqrt(eta^2 + (del2rho^2 * delz^2 * K^2 * M^4 * zd^4) / (12 * dr^2 * fn^2 * (1 + M)^2 * no^2))));

    % del2rho
    Adel2rho(trial_index) = abs((del2rho * delz^2 * K^2 * M^4 * zd^4)/(12 * dr^2 * fn^2 * (1 + M)^2 * no^2 * ...
                            sqrt(eta^2 + (del2rho^2 * delz^2 * K^2 * M^4 * zd^4) / (12 * dr^2 * fn^2 * (1 + M)^2 * no^2))));

end

%% calculate expected values of sensitivity factors

Aetae = median(Aeta, 'omitnan');
AMe = median(AM, 'omitnan');
Azde = median(Azd, 'omitnan');
Afne = median(Afn, 'omitnan');
Adelze = median(Adelz, 'omitnan');
Adel2rhoe = median(Adel2rho, 'omitnan');

%% display expected value to the user

fid = fopen(fullfile(results_save_directory, 'coefficent-statistics-diameter.txt'), 'w');
fprintf(fid, 'Expected Values\n');
fprintf(fid, 'eta: %.2g, %.2g\n', Aetae, Aetae * deta);
fprintf(fid, 'M: %.2g, %.2g\n', AMe, AMe * dM);
fprintf(fid, 'zd: %.2g, %.2g\n', Azde, Azde * dzd);
fprintf(fid, 'fn: %.2g, %.2g\n', Afne, Afne * dfn);
fprintf(fid, 'delz: %.2g, %.2g\n', Adelze, Adelze * ddelz);
fprintf(fid, 'del2rho: %.2g, %.2g\n', Adel2rhoe, Adel2rhoe * ddel2rho);

%% calculate pdf of squares of sensitivity factors

[N_eta, edges_eta] = histcounts(Aeta, num_bins, 'Normalization', 'pdf');
[N_M, edges_M] = histcounts(AM, num_bins, 'Normalization', 'pdf');
[N_zd, edges_zd] = histcounts(Azd, num_bins, 'Normalization', 'pdf');
[N_fn, edges_fn] = histcounts(Afn, num_bins, 'Normalization', 'pdf');
[N_delz, edges_delz] = histcounts(Adelz, num_bins, 'Normalization', 'pdf');
[N_del2rho, edges_del2rho] = histcounts(Adel2rho, num_bins, 'Normalization', 'pdf');

%% calculate pdf of variance contributions

[N_eta_v, edges_eta_v] = histcounts(Aeta * deta, num_bins, 'Normalization', 'pdf');
[N_M_v, edges_M_v] = histcounts(AM * dM, num_bins, 'Normalization', 'pdf');
[N_zd_v, edges_zd_v] = histcounts(Azd * dzd, num_bins, 'Normalization', 'pdf');
[N_fn_v, edges_fn_v] = histcounts(Afn * dfn, num_bins, 'Normalization', 'pdf');
[N_delz_v, edges_delz_v] = histcounts(Adelz * ddelz, num_bins, 'Normalization', 'pdf');
[N_del2rho_v, edges_del2rho_v] = histcounts(Adel2rho * ddel2rho, num_bins, 'Normalization', 'pdf');


%% plot pdf of sensitivity factors

figure
subplot(1, 2, 1)
plot(edges_eta(1:end-1), N_eta);
hold on
plot(edges_M(1:end-1), N_M);
plot(edges_zd(1:end-1), N_zd);
plot(edges_fn(1:end-1), N_fn);
plot(edges_delz(1:end-1), N_delz);
plot(edges_del2rho(1:end-1), N_del2rho);


xlim([0 1])
ylim([0 10])

title('PDF of |\partial \eta_X/\partial f|')
legend('\eta', 'M', 'z_D', 'f#', '\Delta z', '\nabla^2 \rho')

%% plot pdf of variance contributions

subplot(1, 2, 2)
plot(edges_eta_v(1:end-1), N_eta_v);
hold on
plot(edges_M_v(1:end-1), N_M_v);
plot(edges_zd_v(1:end-1), N_zd_v);
plot(edges_fn_v(1:end-1), N_fn_v);
plot(edges_delz_v(1:end-1), N_delz_v);
plot(edges_del2rho_v(1:end-1), N_del2rho_v);

xlim([0 1])
ylim([0 10])

title('PDF of |\partial \eta_X/\partial f| \Delta f')
legend('\eta', 'M', 'z_D', 'f#', '\Delta z', '\nabla^2 \rho')

set(gcf, 'Position', [465   502   750   300])
set(gcf, 'Resize', 'off');

if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'sensitivity_analysis_mc_diameter', [1, 1, 1]);
end

%% make violin plots of pdfs of contributions

figure
violinplot([Aeta'*deta, AM'*dM, Azd'*dzd, Afn'*dfn, Adelz'*ddelz, Adel2rho'*ddel2rho], {'\eta'; 'M'; 'z_D'; 'f#'; '\Delta z'; '\nabla^2 \rho'}, ...
    'showdata', false, 'shownotches', true, 'edgecolor', [1, 1, 1], 'ViolinAlpha', 0.5);
ylim([0 1])
title('PDF of (\partial \eta_X/\partial f) (\Delta f)')

if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'sensitivity_analysis_mc_diameter_contrib_violin', [1, 1, 0]);
end
