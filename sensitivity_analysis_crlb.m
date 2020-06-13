clear
close all
clc

restoredefaultpath;
addpath(genpath('/scratch/shannon/c/aether/Projects/BOS/general-codes/matlab-codes/'));
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

sne = 5;
alphaoe = 1000;
etaxe = 3/4;
etaye = 3/4;
Re = 0;

%% range for each parameter

dsn = 2*3;
dalphao = 2*500;
detax = 2*2/4;
detay = 2*2/4;
dR = 0.99;

%%

Asn = nans(1, num_trials);
Aalphao = nans(1, num_trials);
Aetax = nans(1, num_trials);
Aetay = nans(1, num_trials);
AR = nans(1, num_trials);

%%

parfor trial_index = 1:num_trials
    
    % generate random numbers
    rand_nums = rand(1, 5) - 0.5;
    
    %% calculate parameter values
    
    sn = sne + rand_nums(1) * dsn;
    alphao = alphaoe + rand_nums(2) * dalphao;
    etax = etaxe + (rand_nums(3) + 0.5) * detax;
    etay = etaye + (rand_nums(4) + 0.5) * detay;
    R = Re + (rand_nums(5) + 0.5) * dR;
    
    %% calculate sensitivity coefficient
    
    Asn(trial_index) = abs((2 * etax^(3/2) * sqrt(etay) * sqrt(2*pi) * (1 - R^2)^(1/4)) / alphao);
    Aalphao(trial_index) = abs(-((2 * etax^(3/2) *sqrt(etay) *sqrt(2*pi) * (1 - R^2)^(1/4) * sn) / alphao^2));
    Aetax(trial_index) = abs((3 * sqrt(etax) * sqrt(etay) * sqrt(2*pi) * (1 - R^2)^(1/4) * sn) / alphao);
    Aetay(trial_index) = abs((etax^(3/2) * sqrt(2*pi) * (1 - R^2)^(1/4) * sn) / (alphao * sqrt(etay)));
    AR(trial_index) = abs(-((etax^(3/2) * sqrt(etay) * sqrt(2*pi)* R * sn) / (alphao * (1 - R^2)^(3/4))));
    
end

%% calculate expected values of sensitivity factors

Asne = median(Asn, 'omitnan');
Aalphaoe = median(Aalphao, 'omitnan');
Aetaxe = median(Aetax, 'omitnan');
Aetaye = median(Aetay, 'omitnan');
ARe = median(AR, 'omitnan');

%% display expected value to the user

fid = fopen(fullfile(results_save_directory, 'coefficent-statistics-crlb.txt'), 'w');
fprintf(fid, 'Median Values\n');
fprintf(fid, 'sn: %.2g, %.2g\n', Asne, Asne * dsn);
fprintf(fid, 'alphao: %.2g, %.2g\n', Aalphaoe, Aalphaoe * dalphao);
fprintf(fid, 'etax: %.2g, %.2g\n', Aetaxe, Aetaxe * detax);
fprintf(fid, 'etay: %.2g, %.2g\n', Aetaye, Aetaye * detay);
fprintf(fid, 'R: %.2g, %.2g\n', ARe, ARe * dR);

%% calculate pdf of sensitivity factors

[N_sn, edges_sn] = histcounts(Asn, num_bins, 'Normalization', 'pdf');
[N_alphao, edges_alphao] = histcounts(Aalphao, num_bins, 'Normalization', 'pdf');
[N_etax, edges_etax] = histcounts(Aetax, num_bins, 'Normalization', 'pdf');
[N_etay, edges_etay] = histcounts(Aetay, num_bins, 'Normalization', 'pdf');
[N_R, edges_R] = histcounts(AR, num_bins, 'Normalization', 'pdf');

%% calculate pdf of standard deviation contributions

[N_sn_v, edges_sn_v] = histcounts(Asn * dsn, num_bins, 'Normalization', 'pdf');
[N_alphao_v, edges_alphao_v] = histcounts(Aalphao * dalphao, num_bins, 'Normalization', 'pdf');
[N_etax_v, edges_etax_v] = histcounts(Aetax * detax, num_bins, 'Normalization', 'pdf');
[N_etay_v, edges_etay_v] = histcounts(Aetay * detay, num_bins, 'Normalization', 'pdf');
[N_R_v, edges_R_v] = histcounts(AR * dR, num_bins, 'Normalization', 'pdf');


%% plot pdf of squares of sensitivity factors

figure
subplot(1, 2, 1)
plot(edges_sn(1:end-1), N_sn);
hold on
plot(edges_alphao(1:end-1), N_alphao);
plot(edges_etax(1:end-1), N_etax);
plot(edges_etay(1:end-1), N_etay);
plot(edges_R(1:end-1), N_R);

xlim([0 0.05])
ylim([0 300])

title('PDF of (\partial \sigma_X/\partial f)')
legend('\sigma_n', '\alpha_0', '\eta_x', '\eta_y', 'R')

% set(gcf, 'Position', [465   502   631   354])
% set(gcf, 'Resize', 'off');

% %% plot pdf of variance contributions

subplot(1, 2, 2)
plot(edges_sn_v(1:end-1), N_sn_v);
hold on
plot(edges_alphao_v(1:end-1), N_alphao_v);
plot(edges_etax_v(1:end-1), N_etax_v);
plot(edges_etay_v(1:end-1), N_etay_v);
plot(edges_R_v(1:end-1), N_R_v);

xlim([0 0.1])
ylim([0 75])

title('PDF of (\partial \sigma_X/\partial f) (\Delta f)')
legend('\sigma_n', '\alpha_0', '\eta_x', '\eta_y', 'R')

set(gcf, 'Position', [465   502   750   300])
set(gcf, 'Resize', 'off');

if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'sensitivity_analysis_mc_crlb', [1, 1, 1]);
end

%% make violin plots of pdfs of contributions

figure
violinplot([Asn'*dsn, Aalphao'*dalphao, Aetax'*detax, Aetay'*detay, AR'*dR], {'\sigma_n'; '\alpha_0'; '\eta_x'; '\eta_y'; 'R'}, ...
    'showdata', false, 'shownotches', true, 'edgecolor', [1, 1, 1], 'ViolinAlpha', 0.5);
ylim([0 0.1])
title('PDF of (\partial \sigma_X/\partial f) (\Delta f)')

if save_figures
    save_figure_to_png_eps_fig(figure_save_directory, 'sensitivity_analysis_mc_crlb_contrib_violin', [1, 1, 0]);
end