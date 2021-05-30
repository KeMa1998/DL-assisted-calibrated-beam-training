clear all;
close all;
clc

%plot different output narrow beam number
figure;
hold on;
xlabel('Candidate narrow beam number $N_{\rm{Tx}}$', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
grid on;
set(gca,'xscale','log');
xlim([32 256]);
xticks([32 64 128 256]);
set(gca,'xticklabel',[32 64 128 256]);


power_ratios = [];
beamnum = [32; 64; 128; 256];
for i = 1 : length(beamnum)
    beam = beamnum(i);
    file = ['baseline1_evaluation_' num2str(beam) 'beam.mat'];
    load(file);
    power_ratios = [power_ratios mean(power_ratio(6 : 10))];
end
plot([32; 64; 128; 256], power_ratios, 'r-o', 'LineWidth', 1.5);

power_ratio = [];
beamnum = [32; 64; 128; 256];
for i = 1 : length(beamnum)
    beam = beamnum(i);
    file = ['TCOM_LOS_' num2str(beam) 'beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_baseline2_v1_15dBm_evaluation.mat'];
    load(file);
    power_ratio = [power_ratio mean(mean(BL_eval(6 : 10, 1, :)))];
end
plot([32; 64; 128; 256], power_ratio, 'b-*', 'LineWidth', 1.5);

power_ratios = [];
beamnum = [32; 64; 128; 256];
for i = 1 : length(beamnum)
    beam = beamnum(i);
    file = ['baseline3_evaluation_' num2str(beam) 'beam.mat'];
    load(file);
    power_ratios = [power_ratios mean(power_ratio(6 : 10))];
end
plot([32; 64; 128; 256], power_ratios, '+-', 'color', [0 0.5 0.5], 'LineWidth', 1.5);

power_ratio = [];
beamnum = [32; 64; 128; 256];
for i = 1 : length(beamnum)
    beam = beamnum(i);
    file = ['TCOM_LOS_' num2str(beam) 'beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_proposed1_v1_15dBm_evaluation.mat'];
    load(file);
    power_ratio = [power_ratio mean(mean(BL_eval(6 : 10, 1, :)))];
end
plot([32; 64; 128; 256], power_ratio, 'g-v', 'LineWidth', 1.5);

power_ratio = [];
beamnum = [32; 64; 128; 256];
for i = 1 : length(beamnum)
    beam = beamnum(i);
    file = ['TCOM_LOS_' num2str(beam) 'beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed2_v1_15dBm_evaluation.mat'];
    load(file);
    power_ratio = [power_ratio mean(mean(BL_eval(6 : 10, 1, :)))];
end
plot([32; 64; 128; 256], power_ratio, 'm-^', 'LineWidth', 1.5);

power_ratio = [];
beamnum = [32; 64; 128; 256];
for i = 1 : length(beamnum)
    beam = beamnum(i);
    file = ['TCOM_LOS_' num2str(beam) 'beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=5_evaluation'];
    load(file);
    power_ratio = [power_ratio mean(mean(BL_eval(6 : 10, 1, :)))];
end
plot([32; 64; 128; 256], power_ratio, 'p-', 'color', [0.5 0.5 0], 'LineWidth', 1.5);

power_ratio = [];
beamnum = [32; 64; 128; 256];
for i = 1 : length(beamnum)
    beam = beamnum(i);
    file = ['TCOM_LOS_' num2str(beam) 'beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_v1_k=7_evaluation'];
    load(file);
    power_ratio = [power_ratio mean(mean(BL_eval(6 : 10, 1, :)))];
end
plot([32; 64; 128; 256], power_ratio, 'x-', 'color', [0.5 0 0.5], 'LineWidth', 1.5);

legend('Baseline 1 [14]', 'Baseline 2 [25]', 'Baseline 3 [16]', 'CNN assisted CBT of Sec. III',...
    'LSTM assisted CBT of Sec. IV', 'Enhanced adaptive CBT of Sec. V (ONC, $K=5$)', 'Enhanced adaptive CBT of Sec. V (ONC, $K=7$)',...
    'interpreter', 'latex');
ylim([0.1 0.9]);