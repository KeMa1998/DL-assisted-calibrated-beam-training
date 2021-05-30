clear all;
close all;
clc

%% Investigation of learning rate
initial_loss = - log(1 / 64);

%CNN assisted
figure;
grid on;
hold on;
load('..\training_parameter\learning_rate\proposed1_lr0.003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'r-', 'LineWidth', 1.5, 'markersize', 2);

load('..\training_parameter\learning_rate\proposed1_lr0.001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'b-.', 'LineWidth', 1.5, 'markersize', 2.5);

load('..\training_parameter\learning_rate\proposed1_lr0.0003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'g:', 'LineWidth', 1.5, 'markersize', 2);

load('..\training_parameter\learning_rate\proposed1_lr0.0001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'm--', 'LineWidth', 1.5, 'markersize', 2);

legend('$r_{\rm{L}}$ = 0.003', '$r_{\rm{L}}$ = 0.001', '$r_{\rm{L}}$ = 0.0003', '$r_{\rm{L}}$ = 0.0001', 'interpreter', 'latex');
xlabel('Epoch', 'interpreter', 'latex');
ylabel('${\rm{loss}}_{\rm{n}}$', 'interpreter', 'latex');
ylim([1 4.5]);

%LSTM assisted
figure;
grid on;
hold on;
load('..\training_parameter\learning_rate\proposed2_lr0.003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'r-', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed2_lr0.001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'b-.', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed2_lr0.0003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'g:', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed2_lr0.0001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'm--', 'LineWidth', 1.5);

legend('$r_{\rm{L}}$ = 0.003', '$r_{\rm{L}}$ = 0.001', '$r_{\rm{L}}$ = 0.0003', '$r_{\rm{L}}$ = 0.0001', 'interpreter', 'latex');
xlabel('Epoch', 'interpreter', 'latex');
ylabel('${\rm{loss}}_{\rm{n}}$', 'interpreter', 'latex');
ylim([1 4.5]);

%Adaptive, K=7
figure;
grid on;
hold on;
load('..\training_parameter\learning_rate\proposed3(basic)_ONC_k=7_lr0.001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'r-', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(basic)_ONC_k=7_lr0.0003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'b-', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(basic)_ONC_k=7_lr0.0001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'g-', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(basic)_ONC_k=7_lr0.00003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'm-', 'LineWidth', 1.5);

%
load('..\training_parameter\learning_rate\proposed3(basic)_MPC_k=7_lr0.001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'r--', 'LineWidth', 1.5, 'markersize', 2);

load('..\training_parameter\learning_rate\proposed3(basic)_MPC_k=7_lr0.0003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'b--', 'LineWidth', 1.5, 'markersize', 2);

load('..\training_parameter\learning_rate\proposed3(basic)_MPC_k=7_lr0.0001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'g--', 'LineWidth', 1.5, 'markersize', 2);

load('..\training_parameter\learning_rate\proposed3(basic)_MPC_k=7_lr0.00003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'm--', 'LineWidth', 1.5, 'markersize', 2);

legend('ONC, $r_{\rm{L}}$ = 0.001', 'ONC, $r_{\rm{L}}$ = 0.0003', 'ONC, $r_{\rm{L}}$ = 0.0001', 'ONC, $r_{\rm{L}}$ = 0.00003',...
    'MPC, $r_{\rm{L}}$ = 0.001', 'MPC, $r_{\rm{L}}$ = 0.0003', 'MPC, $r_{\rm{L}}$ = 0.0001', 'MPC, $r_{\rm{L}}$ = 0.00003',...
    'interpreter', 'latex', 'fontsize', 7);
xlabel('Epoch', 'interpreter', 'latex');
ylabel('${\rm{loss}}_{\rm{n}}$', 'interpreter', 'latex');
ylim([1 4.5]);

%Enhanced adaptive, K=7
figure;
grid on;
hold on;
load('..\training_parameter\learning_rate\proposed3(enhanced)_ONC_k=7_lr0.001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'r-', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(enhanced)_ONC_k=7_lr0.0003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'b-', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(enhanced)_ONC_k=7_lr0.0001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'g-', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(enhanced)_ONC_k=7_lr0.00003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'm-', 'LineWidth', 1.5);

%
load('..\training_parameter\learning_rate\proposed3(enhanced)_MPC_k=7_lr0.001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'r--', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(enhanced)_MPC_k=7_lr0.0003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'b--', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(enhanced)_MPC_k=7_lr0.0001.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'g--', 'LineWidth', 1.5);

load('..\training_parameter\learning_rate\proposed3(enhanced)_MPC_k=7_lr0.00003.mat');
plot(0 : 80, [initial_loss; squeeze(mean(loss_eval, 2)) / 10], 'm--', 'LineWidth', 1.5);

legend('ONC, $r_{\rm{L}}$ = 0.001', 'ONC, $r_{\rm{L}}$ = 0.0003', 'ONC, $r_{\rm{L}}$ = 0.0001', 'ONC, $r_{\rm{L}}$ = 0.00003',...
    'MPC, $r_{\rm{L}}$ = 0.001', 'MPC, $r_{\rm{L}}$ = 0.0003', 'MPC, $r_{\rm{L}}$ = 0.0001', 'MPC, $r_{\rm{L}}$ = 0.00003',...
    'interpreter', 'latex', 'fontsize', 7);
xlabel('Epoch', 'interpreter', 'latex');
ylabel('${\rm{loss}}_{\rm{n}}$', 'interpreter', 'latex');
ylim([1 4.5]);


%%
% FLOPs
flop_CNN = 2 * (2 * 64 * 3 * 6) + 2 * (64 * 256 * 3 * 3);
flop_LSTM1 = 2 * (256 * 256 * 8 + 256 * (256 + 20));
flop_LSTM2 = flop_LSTM1;
flop_fc1 = 2 * 256 * 64;
flop_fc2 = 2 * 256 * 16;

flop_p1 = flop_CNN + flop_fc1;
flop_p2 = flop_p1 + flop_LSTM1;
flop_p3 = flop_p2 + flop_LSTM2 + flop_fc2; % enhanced

%%
% Investigation of the impact of mu
figure;
hold on;
grid on;
loss_sery = zeros(25, 1);
count = 0;
for mu = 0.1 : 0.1 : 2.5
    count = count + 1;
    load(['..\training_parameter\mu\TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_K=7_mu' num2str(mu) '.mat']);
    loss_sery(count) = mean(loss_eval(80, 1 : 5)) / 10;
end
plot(0.1 : 0.1 : 2.5, loss_sery, 'r-o', 'LineWidth', 1.5);

loss_sery = zeros(10, 1);
count = 0;
for mu = 0.1 : 0.1 : 2.5
    count = count + 1;
    load(['..\training_parameter\mu\TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_MPC_K=7_mu' num2str(mu) '.mat']);
    loss_sery(count) = mean(loss_eval(80, 1 : 5)) / 10;
end
plot(0.1 : 0.1 : 2.5, loss_sery, 'b-*', 'LineWidth', 1.5);

xlim([0 2.5]);
ylim([1.16 1.21]);
xlabel('Weight coefficient $\mu$', 'interpreter', 'latex');
ylabel('${\rm{loss}}_{\rm{n}}$', 'interpreter', 'latex');
legend('ONC', 'MPC', 'interpreter', 'latex');

%%
% Execution time
load('..\complexity\TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_K=7_execution_time.mat');
batch_size = 16;
time_FC1 = mean(times_FC1) / batch_size;
time_FC2 = mean(times_FC2) / batch_size;
time_CNN = mean(times_CNN) / batch_size;
time_LSTM1 = mean(times_LSTM1) / batch_size;
time_LSTM2 = mean(times_LSTM2) / batch_size;

time_p1 = time_CNN + time_FC1;
time_p2 = time_CNN + time_LSTM1 + time_FC1;
time_p3 = time_CNN + time_LSTM1 + time_LSTM2 + time_FC1 + time_FC2; % enhanced

%%
%Training time per epoch
load('..\complexity\TCOM_LOS_64beam_2CNN_0LSTM_256feature_16Tx_RK=8dB_proposed1_testtime.mat');
average_time_CNN = mean(mean(times));

load('..\complexity\TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed2_testtime.mat');
average_time_LSTM = mean(mean(times));

load('..\complexity\TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3(basic)_k=7_testtime.mat');
average_time_adaptive = mean(mean(times));

load('..\complexity\TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3(enhanced)_k=7_testtime.mat');
average_time_enhanced_adaptive = mean(mean(times));