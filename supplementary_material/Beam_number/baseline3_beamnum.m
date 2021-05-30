clear all;
close all;
clc

%Baseline 3
% Chiu et al., Active Learning and CSI Acquisition for mmWwave Initial Alignment
rng(1)

N = 256;
num = 256;

% BS narrow beam number
B = 256;
% BS antenna number
rx = 64;
resolution = 256; % targeted resolution
measure_t = 16; % number of each beam training
layer = log2(min(resolution, rx));

sector_start=-pi/2; % Sectoar start point
sector_end=pi/2; % Sector end point
% calculate candidate beams
candidate_beam_angle = sector_start + (sector_end - sector_start) / B * [0.5 : 1 : B - 0.5];
candidate_beam = sin(candidate_beam_angle(end : -1 : 1));
candidate_beam = exp(-1i * pi * candidate_beam' * [0 : rx - 1]);

power_ratio = zeros(10, 1, 5);
pdf = zeros(101, 10, 1);
noise_e = 112; % Equivalent noise including AWGN and NLOS components
count = 0;

for snr_e = 120 % Equivalent SNR calculated as (P - sigma^2)
    count = count + 1;
    for n = 1 : 16
        file_name = ['..\dataset\testing_15dBm_channel\data_TCOM(withLOSparameter)_16Tx_64Tx_RK8dB_' num2str(n) '.mat'];
        load(file_name);
        for i = 1 : 256
            for j = 1 : 10
                mm_channel = squeeze(channel_m(i, j, :));
                LOS_parameter = LOS_parameters(i, j);
                
                for bbb = 1 : B
                    beam_select(bbb) = candidate_beam(bbb, :) * mm_channel;
                end
                rsrp = abs(beam_select);
                [~, max_rsrp_location] = max(rsrp);
                
                pis = 1 / resolution * ones(resolution, 1); % initialize pi
                for t = 1 : measure_t % for t = 1, 2, ..., measure_t
                    k = 0;
                    l_star = 1;
                    for l = 1 : layer % for l = 1, 2, ..., S
                        pi_test = zeros(2 ^ l, 1);
                        for test_num = 1 : 2 ^ l
                            pi_test(test_num) = sum(pis(1 + (resolution / (2 ^ l)) * (test_num - 1) : (resolution / (2 ^ l)) * test_num));
                        end
                        [max_pi, location] = max(pi_test);
                        % if l == layer, no descendent is considered
                        if(l == layer)
                            l_final = l;
                            k_final = location - 1;
                            break;
                        % when max_pi > 0.5, select descendent
                        elseif(max_pi > 0.5)
                            l_star = l;
                            descendent1 = sum(pis(1 + (resolution / (2 ^ (l + 1))) * 2 * (location - 1) : (resolution / (2 ^ (l + 1))) * (2 * location - 1)));
                            descendent2 = sum(pis(1 + (resolution / (2 ^ (l + 1))) * (2 * location - 1) : (resolution / (2 ^ (l + 1))) * (2 * location)));
                            if(descendent1 > descendent2)
                                k = 2 * (location - 1);
                            else
                                k = 2 * location - 1;
                            end
                        % else, no descendent is considered
                        else
                            selection1 = sum(pis(1 + (resolution / (2 ^ (l_star))) * floor(k / 2) : (resolution / (2 ^ (l_star))) * (floor(k / 2) + 1)));
                            selection2 = sum(pis(1 + (resolution / (2 ^ (l_star + 1))) * k : (resolution / (2 ^ (l_star + 1))) * (k + 1)));
                            if((abs(selection1 - 0.5) > abs(selection2 - 0.5)))
                                l_final = l_star + 1;
                                k_final = k;
                            else
                                l_final = l_star;
                                k_final = floor(k / 2);
                            end
                            break;
                        end
                    end
                    
                    % formulate the adopted beam
                    w = exp(-1i * pi * sin(sector_start + (sector_end - sector_start) / (2 ^ l_final) * (0.5 + k_final)) * [0 : 2 ^ l - 1]) / sqrt(2 ^ l);
                    
                    % measure the received signal
                    y = w * mm_channel(1 : 2 ^ l);
                    y = awgn(y, snr_e);
                    
                    % formulate candidate beams
                    candidate_beam_angle0 = sector_start + (sector_end - sector_start) / resolution * [0.5 : 1 : resolution - 0.5];
                    candidate_beam0 = sin(candidate_beam_angle0);
                    candidate_beam0 = exp(-1i * pi * candidate_beam0' * [0 : 2^l - 1]);
                    
                    % calculate the posterior probability
                    f = exp(- (abs(y - LOS_parameter * w * candidate_beam0.')).^2 / 2 / (10^(- noise_e(count) / 10)));
                    
                    % update the posterior probability
                    pis_new = zeros(resolution, 1);
                    for r = 1 : resolution
                        pis_new(r) = pis(r) * f(r) / (sum(pis([1 : r - 1, r + 1 : end]) .* f([1 : r - 1, r + 1 : end])') + 1e-16);
                    end
                    pis = pis_new;
                    pis = pis / sum(pis);
                end
                [~, max_location] = max(pis);
                [~, sort_index] = sort(pis,'descend');
                % calculate top-5 beamforming gain
                power_ratio(j, count, 1) = power_ratio(j, count, 1) + rsrp(max_location)^2 / max(rsrp)^2;
                power_ratio(j, count, 2) = power_ratio(j, count, 2) + max(rsrp(sort_index(1 : 2)))^2 / max(rsrp)^2;
                power_ratio(j, count, 3) = power_ratio(j, count, 3) + max(rsrp(sort_index(1 : 3)))^2 / max(rsrp)^2;
                power_ratio(j, count, 4) = power_ratio(j, count, 4) + max(rsrp(sort_index(1 : 4)))^2 / max(rsrp)^2;
                power_ratio(j, count, 5) = power_ratio(j, count, 5) + max(rsrp(sort_index(1 : 5)))^2 / max(rsrp)^2;
                % calculate pdf
                pdf(floor(rsrp(max_location)^2 / max(rsrp)^2 * 100) + 1 : end, j, count) = ...
                    pdf(floor(rsrp(max_location)^2 / max(rsrp)^2 * 100) + 1 : end, j, count) + 1;
            end
        end
    end
end

power_ratio = power_ratio / 256 / 16;
pdf = pdf / 256 / 16;
%save('baseline3_evaluation_256beam.mat', 'power_ratio');