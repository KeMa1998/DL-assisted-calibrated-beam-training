clear all;
close all;
clc;

%% Plot average spectral efficiency
% Specific data can be found in cloud disks
load('averaged_results.mat');
figure;
grid on;
hold on;

% Baseline 1
plot(10 : 10 : 100, ase1, 'r-o', 'linewidth', 1.5);

% Baseline 2
plot(10 : 10 : 100, ase2, 'b-*', 'linewidth', 1.5);

% Baseline 3
plot(10 : 10 : 100, ase3, '+-', 'color', [0 0.5 0.5], 'linewidth', 1.5);

% CNN assisted
plot(10 : 10 : 100, ase_CNN, 'g-v', 'linewidth', 1.5);

% LSTM assisted
plot(10 : 10 : 100, ase_LSTM, 'm-^', 'linewidth', 1.5);

% Enhanced adaptive scheme, ONC, K = 5
plot(10 : 10 : 100, ase_e_adaptive5, 'p-', 'color', [0.5 0.5 0], 'linewidth', 1.5);

% Enhanced adaptive scheme, ONC, K = 7
plot(10 : 10 : 100, ase_e_adaptive7, 'x-', 'color', [0.5 0 0.5], 'linewidth', 1.5);

% LSTM assisted, Kn = 5
plot(10 : 10 : 100, ase_LSTM_a, 'm-.^', 'linewidth', 1.5);

% Enhanced adaptive scheme, ONC, K = 5, Kn = 5
plot(10 : 10 : 100, ase_e_adaptive5_a, 'p-.', 'color', [0.5 0.5 0], 'linewidth', 1.5);

% Enhanced adaptive scheme, ONC, K = 7, Kn = 5
plot(10 : 10 : 100, ase_e_adaptive7_a, 'x-.', 'color', [0.5 0 0.5], 'linewidth', 1.5);

legend('Baseline 1 [14]', 'Baseline 2 [25]', 'Baseline 3 [16]', 'CNN assisted CBT of Sec. III ($K_{\rm{n}}=0$)',...
    'LSTM assisted CBT of Sec. IV ($K_{\rm{n}}=0$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=5,K_{\rm{n}}=0$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=7,K_{\rm{n}}=0$)',...
    'LSTM assisted CBT of Sec. IV ($K_{\rm{n}}=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=5, K_{\rm{n}}=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=7, K_{\rm{n}}=5$)',...
    'interpreter', 'latex', 'fontsize', 6);
ylim([2 6]);
xlabel('Beam training period $\tau$ (ms)', 'interpreter', 'latex');
ylabel('Average spectral efficiency $\overline{E}$ (bps/Hz)', 'interpreter', 'latex');
