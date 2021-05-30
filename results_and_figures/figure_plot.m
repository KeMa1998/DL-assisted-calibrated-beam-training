clear all;
close all;
clc;


%comparison of adaptive schemes
%K = 5
figure;
hold on;
xlabel('Wide beam training number $t$', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3(basic)_ONC_v1_k=5_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'r-o', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3(basic)_MPC_v1_k=5_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'b-*', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=5_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'g-v', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_MPC_v1_k=5_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'm-^', 'LineWidth', 1.5);

%K = 7
load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3(basic)_ONC_v1_k=7_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'r-.o', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3(basic)_MPC_v1_k=7_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'b-.*', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=7_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'g-.v', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_MPC_v1_k=7_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'm-.^', 'LineWidth', 1.5);

legend('Adaptive CBT of Sec. V (ONC, $K=5$)', 'Adaptive CBT of Sec. V (MPC, $K=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=5$)', 'Enhanced adaptive CBT of Sec. V (MPC, $K=5$)',...
    'Adaptive CBT of Sec. V (ONC, $K=7$)', 'Adaptive CBT of Sec. V (MPC, $K=7$)',...
    'Enhanced adaptive CBT of Sec. V (ONC, $K=7$)', 'Enhanced adaptive CBT of Sec. V (MPC, $K=7$)', 'interpreter', 'latex');

%comparison of K
figure;
hold on;
grid on;
xlabel('Trained wide beam number $K$', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
ONC = [];
for ks = 3 : 9
    load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3(basic)_ONC_v1_k=' num2str(ks) '_evaluation.mat']);
    BL = mean(squeeze(BL_eval(:, 1, :)), 2);
    ONC = [ONC mean(BL(6 : 10))];
end
plot(3 : 9, ONC, 'r-*', 'LineWidth', 1.5);

MPC = [];
for ks = 3 : 9
    load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3(basic)_MPC_v1_k=' num2str(ks) '_evaluation.mat']);
    BL = mean(squeeze(BL_eval(:, 1, :)), 2);
    MPC = [MPC mean(BL(6 : 10))];
end
plot(3 : 9, MPC, 'b-^', 'LineWidth', 1.5);

AONC = [];
for ks = 3 : 2 : 9
    load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=' num2str(ks) '_evaluation.mat']);
    BL = mean(squeeze(BL_eval(:, 1, :)), 2);
    AONC = [AONC mean(BL(6 : 10))];
end
plot(3 : 2 : 9, AONC, 'g-v', 'LineWidth', 1.5);

AMPC = [];
for ks = 3 : 9
    load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_MPC_v1_k=' num2str(ks) '_evaluation.mat']);
    BL = mean(squeeze(BL_eval(:, 1, :)), 2);
    AMPC = [AMPC mean(BL(6 : 10))];
end
plot(3 : 9, AMPC, 'm-o', 'LineWidth', 1.5);
legend('Adaptive CBT of Sec. V (ONC)', 'Adaptive CBT of Sec. V (MPC)', 'Enhanced adaptive CBT of Sec. V (ONC)', 'Enhanced adaptive CBT of Sec. V (MPC)', 'interpreter', 'latex');
ylim([0.55 0.75]);

%Basic comparison
figure;
hold on;
xlabel('Wide beam training number $t$', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;

load('baseline1_evaluation.mat');
BL = power_ratio(2, :, 1);
plot(1 : 10, BL, 'r-o', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_baseline2_v1_15dBm_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'b-*', 'LineWidth', 1.5);

load('baseline3_evaluation.mat');
BL = power_ratio(:, 2, 1);
plot(1 : 10, BL, '-+', 'color', [0 0.5 0.5], 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_proposed1_v1_15dBm_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'g-v', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed2_v1_15dBm_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'm-^', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=5_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'p-', 'color', [0.5 0.5 0], 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=7_evaluation.mat');
BL = mean(squeeze(BL_eval(:, 1, :)), 2);
plot(1 : 10, BL, 'x-', 'color', [0.5 0 0.5], 'LineWidth', 1.5);

legend('Baseline 1 [14]', 'Baseline 2 [25]', 'Baseline 3 [16]', 'CNN assisted CBT of Sec. III', 'LSTM assisted CBT of Sec. IV',...
    'Enhanced adaptive CBT of Sec. V (ONC, $K=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=7$)',...
    'interpreter', 'latex');

%pdf evaluation
figure;
hold on;
xlabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
ylabel('CDF', 'interpreter', 'latex');
grid on;

load('baseline1_evaluation.mat');
pdf = mean(squeeze(pdf(2, 6 : 10, :)), 1);
plot([0 0.01 : 0.0099 : 1], [0 pdf], 'r-.', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_baseline2_v1_15dBm_evaluation.mat');
pdf = mean(squeeze(mean(pdf_eval(6 : 10, :, :), 3)), 1) / 4096;
plot([0 0.01 : 0.0099 : 1], [0, pdf], 'b-.', 'LineWidth', 1.5);

load('baseline3_evaluation.mat');
pdf = mean(squeeze(pdf(:, 6 : 10, 2)), 2);
plot([0 0.01 : 0.0099 : 1], [0 pdf'], '-.', 'color', [0 0.5 0.5], 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_proposed1_v1_15dBm_evaluation.mat');
pdf = mean(squeeze(mean(pdf_eval(6 : 10, :, :), 3)), 1) / 4096;
plot([0 0.01 : 0.0099 : 1], [0 pdf]', 'g-.', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed2_v1_15dBm_evaluation.mat');
pdf = mean(squeeze(mean(pdf_eval(6 : 10, :, :), 3)), 1) / 4096;
plot([0 0.01 : 0.0099 : 1], [0 pdf]', 'm-.', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=5_evaluation.mat');
pdf = mean(squeeze(mean(pdf_eval(6 : 10, :, :), 3)), 1) / 4096;
plot([0 0.01 : 0.0099 : 1], [0 pdf]', '-.', 'color', [0.5 0.5 0], 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=7_evaluation.mat');
pdf = mean(squeeze(mean(pdf_eval(6 : 10, :, :), 3)), 1) / 4096;
plot([0 0.01 : 0.0099 : 1], [0 pdf]', '-.', 'color', [0.5 0 0.5], 'LineWidth', 1.5);

legend('Baseline 1 [14]', 'Baseline 2 [25]', 'Baseline 3 [16]', 'CNN assisted CBT of Sec. III', 'LSTM assisted CBT of Sec. IV',...
    'Enhanced adaptive CBT of Sec. V (ONC, $K=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=7$)',...
    'interpreter', 'latex');

%top-k received power loss
figure;
hold on;
xlabel('Additional trained narrow beam number $K_{\rm{n}}$', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;

load('baseline1_evaluation.mat');
power_ratio = mean(squeeze(power_ratio(2, 6 : 10, :)), 1);
plot(1 : 5, power_ratio, 'ro-', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_baseline2_v1_15dBm_evaluation.mat');
power_ratio = mean(squeeze(mean(BL_eval(6 : 10, 2 : 6, :), 3)), 1);
plot(1 : 5, power_ratio, 'b*-', 'LineWidth', 1.5);

load('baseline3_evaluation.mat');
power_ratio = mean(squeeze(power_ratio(6 : 10, 2, :)), 1);
plot(1 : 5, power_ratio, '-+', 'color', [0 0.5 0.5], 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_proposed1_v1_15dBm_evaluation.mat');
power_ratio = mean(squeeze(mean(BL_eval(6 : 10, :, :), 3)), 1);
plot(1 : 5, power_ratio, 'gv-', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed2_v1_15dBm_evaluation.mat');
power_ratio = mean(squeeze(mean(BL_eval(6 : 10, :, :), 3)), 1);
plot(1 : 5, power_ratio, 'm^-', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=5_evaluation.mat');
power_ratio = mean(squeeze(mean(BL_eval(6 : 10, :, :), 3)), 1);
plot(1 : 5, power_ratio, 'p-', 'color', [0.5 0.5 0], 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=7_evaluation.mat');
power_ratio = mean(squeeze(mean(BL_eval(6 : 10, :, :), 3)), 1);
plot(1 : 5, power_ratio, 'x-', 'color', [0.5 0 0.5], 'LineWidth', 1.5);

legend('Baseline 1 [14]', 'Baseline 2 [25]', 'Baseline 3 [16]', 'CNN assisted CBT of Sec. III', 'LSTM assisted CBT of Sec. IV',...
    'Enhanced adaptive CBT of Sec. V (ONC, $K=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=7$)',...
    'interpreter', 'latex');
set(gca, 'XTick', [1 : 1 : 5]);

%velocity
figure;
hold on;
xlabel('UE velocity $v_{\rm{UE}}$(m/s)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;

load('baseline1_evaluation.mat');
power_ratio = squeeze(power_ratio_v(:, 1));
plot(10 : 10 : 50, power_ratio, 'ro-', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_baseline2_v1_velocity.mat');
power_ratio = mean(squeeze(BL_eval(1, :, :)), 2);
plot(10 : 10 : 50, power_ratio, 'b*-', 'LineWidth', 1.5);

load('baseline3_evaluation.mat');
power_ratio = squeeze(power_ratio_v(1, :));
plot(10 : 10 : 50, power_ratio, '-+', 'color', [0 0.5 0.5], 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_proposed1_v1_velocity.mat');
power_ratio = mean(squeeze(BL_eval(1, :, :)), 2);
plot(10 : 10 : 50, power_ratio, 'gv-', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed2_v1_velocity.mat');
power_ratio = mean(squeeze(mean(squeeze(BL_eval(6 : 10, :, :)), 1)), 2);
plot(10 : 10 : 50, power_ratio, 'm^-', 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=5_velocity.mat');
power_ratio = mean(squeeze(mean(squeeze(BL_eval(6 : 10, :, :)), 1)), 2);
plot(10 : 10 : 50, power_ratio, 'p-', 'color', [0.5 0.5 0], 'LineWidth', 1.5);

load('TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=7_velocity.mat');
power_ratio = mean(squeeze(mean(squeeze(BL_eval(6 : 10, :, :)), 1)), 2);
plot(10 : 10 : 50, power_ratio, 'x-', 'color', [0.5 0 0.5], 'LineWidth', 1.5);

legend('Baseline 1 [14]', 'Baseline 2 [25]', 'Baseline 3 [16]', 'CNN assisted CBT of Sec. III', 'LSTM assisted CBT of Sec. IV',...
    'Enhanced adaptive CBT of Sec. V (ONC, $K=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=7$)',...
    'interpreter', 'latex');
set(gca, 'XTick', [10 : 10 : 50]);

%transmit power
figure;
hold on;
xlabel('Transmit power $P$(dBm)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;

load('baseline1_evaluation.mat');
power_ratio = squeeze(mean(squeeze(power_ratio(:, 6 : 10, 1)), 2));
plot(10 : 5 : 25, power_ratio, 'ro-', 'LineWidth', 1.5);

power_ratios = zeros(4, 5);
count = 0;
for P = 10 : 5 : 25
    count = count + 1;
    load(['TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_baseline2_v1_' num2str(P) 'dBm_evaluation.mat']);
    power_ratios(count, 1) = mean(mean(squeeze(BL_eval(6 : 10, 1, :)), 2));
    load(['TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_proposed1_v1_' num2str(P) 'dBm_evaluation.mat']);
    power_ratios(count, 2) = mean(mean(squeeze(BL_eval(6 : 10, 1, :)), 2));
    load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed2_v1_' num2str(P) 'dBm_evaluation.mat']);
    power_ratios(count, 3) = mean(mean(squeeze(BL_eval(6 : 10, 1, :)), 2));
    if(P == 15)
        load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=5_evaluation.mat']);
        power_ratios(count, 4) = mean(mean(squeeze(BL_eval(6 : 10, 1, :)), 2));
        load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=7_evaluation.mat']);
        power_ratios(count, 5) = mean(mean(squeeze(BL_eval(6 : 10, 1, :)), 2));
    else
        load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=5_' num2str(P) 'dBm_evaluation.mat']);
        power_ratios(count, 4) = mean(mean(squeeze(BL_eval(6 : 10, 1, :)), 2));
        load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=7_' num2str(P) 'dBm_evaluation.mat']);
        power_ratios(count, 5) = mean(mean(squeeze(BL_eval(6 : 10, 1, :)), 2));
    end
end
plot(10 : 5 : 25, power_ratios(:, 1), 'b*-', 'LineWidth', 1.5);

load('baseline3_evaluation.mat');
power_ratio = squeeze(mean(squeeze(power_ratio(6 : 10, :, 1)), 1));
plot(10 : 5 : 25, power_ratio, '-+', 'color', [0 0.5 0.5], 'LineWidth', 1.5);

plot(10 : 5 : 25, power_ratios(:, 2), 'gv-', 'LineWidth', 1.5);
plot(10 : 5 : 25, power_ratios(:, 3), 'm^-', 'LineWidth', 1.5);
plot(10 : 5 : 25, power_ratios(:, 4), 'p-', 'color', [0.5 0.5 0], 'LineWidth', 1.5);
plot(10 : 5 : 25, power_ratios(:, 5), 'x-', 'color', [0.5 0 0.5], 'LineWidth', 1.5);
legend('Baseline 1 [14]', 'Baseline 2 [25]', 'Baseline 3 [16]', 'CNN assisted CBT of Sec. III', 'LSTM assisted CBT of Sec. IV',...
    'Enhanced adaptive CBT of Sec. V (ONC, $K=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=7$)',...
    'interpreter', 'latex');

%the channel power leakage
sector_start=-pi/2;% Sectoar start point
sector_end=pi/2; % Sector end point
B = 5120;
rx = 16;
candidate_beam_angle = sector_start + (sector_end - sector_start) / B * [0.5 : 1 : B - 0.5];
candidate_beam = sin(candidate_beam_angle(end : -1 : 1));
candidate_beam = exp(-1i * pi * candidate_beam' * [0 : rx - 1]);

theta = 0.02 * pi;% LOS AoD
theta = sin(theta);
theta = exp(-1i * pi * theta * [0 : rx - 1]);

result = zeros(B, 1);
for b = 1 : B
    result(b) = candidate_beam(b, :) * theta.';
end
figure;
hold on;
x = - pi / 2 + 0.5 * pi / B : pi / B : pi / 2  - 0.5 * pi / B;
plot(x, abs(result) / 16, 'LineWidth', 1.25);
grid on;
xlim([-pi / 2, pi / 2]);
for b = 1 : rx
    id = B / rx * (b - 0.5);
    plot(x(id) * ones(101, 1), 0 : abs(result(id)) / 100 / 16 : abs(result(id)) / 16, 'LineWidth', 1.25, 'color', [0.75 0.5 0.25]);
    scatter(x(id), abs(result(id)) / 16, 20, [0.75 0.5 0.25], 'filled');
end
xlabel('$\gamma_{\rm{Tx}}$', 'interpreter', 'latex');
ylabel('$|q(\phi_{\rm{LOS}})|$', 'interpreter', 'latex');
set(gca,'xtick',[- pi / 2 : pi / 4 : pi / 2]);
set(gca,'xticklabel',{'-\pi / 2' ,'-\pi / 4', '0', '\pi / 4', '\pi / 2'});

% load(['TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_MPC_v1_K=3_probability.mat']);
% i = 14;
% j = 2;
% figure;
% subplot(2, 1, 1);
% bar(exp(p_eval((i - 1) * 256 + j, :)) / max(exp(p_eval((i - 1) * 256 + j, :))));
% xlabel('Wide beam index', 'interpreter', 'latex');
% ylabel('Normalized predicted probability', 'interpreter', 'latex');
% subplot(2, 1, 2);
% load('..\dataset\testing_15dBm\data_TCOM_16Tx_64Tx_RK8dB_training_69_15dBm.mat');
% bar(squeeze(rsrp_sery_widebeam_m(j, 1, :)).^2, 'facecolor', [0 0.5 0.5]);
% xlabel('Wide beam index', 'interpreter', 'latex');
% ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');