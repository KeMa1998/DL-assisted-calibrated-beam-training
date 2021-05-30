clear all;
close all;
clc;

%Baseline 1
%VTC 2019, Xingyi Luo, Wendong Liu et al.
%Calibrated Beam Training for Millimeter-Wave Massive MIMO Systems

sector_start=-pi/2; % Sectoar start point
sector_end=pi/2; % Sector end point
%establish look-up table
% narrow beam number
n_beam = 64;
% wide beam number
w_beam = 16;
% wide beam antenna number
rx = 16;
% the threshold to select narrow beams
th = zeros(2, n_beam / w_beam * 2 - 1, w_beam);
% 1: left neighboring beam; 2: right neighboring beam
for i = 1 : 2
    % divide wide beams by (n_beam / w_beam * 2)
    for j = 1 : n_beam / w_beam * 2 - 1
        %generate threshold
        for k = 1 : w_beam
            los_angle = sector_start + ((k - 1) * (n_beam / w_beam) + 0.5 * j) / n_beam * (sector_end - sector_start);
            channel = sin(los_angle);
            channel = exp(-1i * pi * channel * [0 : rx - 1]);
            beam_angle1 = - (sector_start + (k - 0.5) / w_beam * (sector_end - sector_start));
            candidate_beam1 = sin(beam_angle1);
            candidate_beam1 = exp(-1i * pi * candidate_beam1 * [0 : rx - 1]);
            if (k == 1 && i == 1)
                beam_angle2 = - (sector_start + (16 - 0.5) / w_beam * (sector_end - sector_start));
                candidate_beam2 = sin(beam_angle2);
                candidate_beam2 = exp(-1i * pi * candidate_beam2 * [0 : rx - 1]);
            elseif (k == w_beam && i == 2)
                beam_angle2 = - (sector_start + (1 - 0.5) / w_beam * (sector_end - sector_start));
                candidate_beam2 = sin(beam_angle2);
                candidate_beam2 = exp(-1i * pi * candidate_beam2 * [0 : rx - 1]);
            elseif (i == 1)
                beam_angle2 = - (sector_start + (k - 1 - 0.5) / w_beam * (sector_end - sector_start));
                candidate_beam2 = sin(beam_angle2);
                candidate_beam2 = exp(-1i * pi * candidate_beam2 * [0 : rx - 1]);
            else
                beam_angle2 = - (sector_start + (k + 1 - 0.5)  / w_beam * (sector_end - sector_start));
                candidate_beam2 = sin(beam_angle2);
                candidate_beam2 = exp(-1i * pi * candidate_beam2 * [0 : rx - 1]);
            end
            %save threshold
            th(i, j, k) = abs(candidate_beam1 * channel.') / abs(candidate_beam2 * channel.');
        end
    end
end

%different transmit power
% save normalized beamforming gain
power_ratio = zeros(4, 10, 5);
% save CDF
pdf = zeros(4, 10, 101);
count = 0;
% for different transmit power P
for P = 10 : 5 : 25
    count = count + 1;
    for i = 65 : 80
        load(['../../dataset/testing_' num2str(P) 'dBm/data_TCOM_16Tx_64Tx_RK8dB_training_'...
            num2str(i) '_' num2str(P) 'dBm.mat']);
        for j = 1 : 256
            for k = 1 : 10
                % select the optimal narrow beam
                % step 1: select the applied neighboring beam
                % step 2: calculate the range of the LOS AoD
                % step 3: select the optimal narrow beam in the optimal
                % wide beam according to the LOS AoD
                % step 4: select the optimal narrow beam beyond the optimal
                % wide beam according to the LOS AoD
                [~, max_location] = max(squeeze(rsrp_sery_widebeam_m(j, k, :)));
                if(max_location == 1)
                    neibor_beam1 = rsrp_sery_widebeam_m(j, k, 16);
                else
                    neibor_beam1 = rsrp_sery_widebeam_m(j, k, max_location - 1);
                end
                if(max_location == 16)
                    neibor_beam2 = rsrp_sery_widebeam_m(j, k, 1);
                else
                    neibor_beam2 = rsrp_sery_widebeam_m(j, k, max_location + 1);
                end
                if(neibor_beam1 > neibor_beam2)
                    if(neibor_beam1 > (1 / th(1, 2, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 1;
                        power_ratio(count, k, 1) = power_ratio(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                        power_ratio(count, k, 2) = power_ratio(count, k, 2) + max(rsrp_sery_no_noise_m(j, k,...
                            mod([optimal : optimal + 1] - 1, n_beam) + 1))^2;
                        power_ratio(count, k, 3) = power_ratio(count, k, 3) + max(rsrp_sery_no_noise_m(j, k,...
                            mod([optimal : optimal + 2] - 1, n_beam) + 1))^2;
                        power_ratio(count, k, 4) = power_ratio(count, k, 4) + max(rsrp_sery_no_noise_m(j, k,...
                            mod([optimal : optimal + 3] - 1, n_beam) + 1))^2;
                        power_ratio(count, k, 5) = power_ratio(count, k, 5) + max(rsrp_sery_no_noise_m(j, k,...
                            mod([optimal - 1 : optimal + 3] - 1, n_beam) + 1))^2;
                    else
                        optimal = (max_location - 1) * n_beam / w_beam + 2;
                        if(neibor_beam1 > (1 / th(1, 3, max_location)))
                            power_ratio(count, k, 1) = power_ratio(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                            power_ratio(count, k, 2) = power_ratio(count, k, 2) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 1 : optimal] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 3) = power_ratio(count, k, 3) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 1 : optimal + 1] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 4) = power_ratio(count, k, 4) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 1 : optimal + 2] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 5) = power_ratio(count, k, 5) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 2 : optimal + 2] - 1, n_beam) + 1))^2;
                        else
                            power_ratio(count, k, 1) = power_ratio(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                            power_ratio(count, k, 2) = power_ratio(count, k, 2) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal : optimal + 1] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 3) = power_ratio(count, k, 3) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 1 : optimal + 1] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 4) = power_ratio(count, k, 4) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 1 : optimal + 2] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 5) = power_ratio(count, k, 5) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 2 : optimal + 2] - 1, n_beam) + 1))^2;
                        end
                    end
                else
                    if(neibor_beam2 > (1 / th(2, 6, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 4;
                        power_ratio(count, k, 1) = power_ratio(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                        power_ratio(count, k, 2) = power_ratio(count, k, 2) + max(rsrp_sery_no_noise_m(j, k,...
                            mod([optimal - 1 : optimal] - 1, n_beam) + 1))^2;
                        power_ratio(count, k, 3) = power_ratio(count, k, 3) + max(rsrp_sery_no_noise_m(j, k,...
                            mod([optimal - 2 : optimal] - 1, n_beam) + 1))^2;
                        power_ratio(count, k, 4) = power_ratio(count, k, 4) + max(rsrp_sery_no_noise_m(j, k,...
                            mod([optimal - 3 : optimal] - 1, n_beam) + 1))^2;
                        power_ratio(count, k, 5) = power_ratio(count, k, 5) + max(rsrp_sery_no_noise_m(j, k,...
                            mod([optimal - 3 : optimal + 1] - 1, n_beam) + 1))^2;
                    else
                        optimal = (max_location - 1) * n_beam / w_beam + 3;
                        if(neibor_beam2 < (1 / th(2, 5, max_location)))
                            power_ratio(count, k, 1) = power_ratio(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                            power_ratio(count, k, 2) = power_ratio(count, k, 2) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 1 : optimal] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 3) = power_ratio(count, k, 3) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 1 : optimal + 1] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 4) = power_ratio(count, k, 4) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 2 : optimal + 1] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 5) = power_ratio(count, k, 5) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 2 : optimal + 2] - 1, n_beam) + 1))^2;
                        else
                            power_ratio(count, k, 1) = power_ratio(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                            power_ratio(count, k, 2) = power_ratio(count, k, 2) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal : optimal + 1] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 3) = power_ratio(count, k, 3) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 1 : optimal + 1] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 4) = power_ratio(count, k, 4) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 2 : optimal + 1] - 1, n_beam) + 1))^2;
                            power_ratio(count, k, 5) = power_ratio(count, k, 5) + max(rsrp_sery_no_noise_m(j, k,...
                                mod([optimal - 2 : optimal + 2] - 1, n_beam) + 1))^2;
                        end
                    end
                end
                pdf(count, k, floor(rsrp_sery_no_noise_m(j, k, optimal)^2 * 100 + 1) : end) = ...
                    pdf(count, k, floor(rsrp_sery_no_noise_m(j, k, optimal)^2 * 100) + 1 : end) + 1;
            end
        end
    end
end
power_ratio = power_ratio / 256 / 16;
pdf = pdf / 256 / 16;

% different velocity
% similar to the former codes
power_ratio_v = zeros(5, 10);
count = 0;
for v = 10 : 10 : 50
    count = count + 1;
    for i = 1 : 32
        load(['../../dataset/velocity_' num2str(v) '/data_TCOM_16Tx_64Tx_speed' num2str(v) '_RK8dB_training_'...
            num2str(i) '_15dBm.mat']);
        for j = 1 : 256
            for k = 1 : 10
                [~, max_location] = max(squeeze(rsrp_sery_widebeam_m(j, k, :)));
                if(max_location == 1)
                    neibor_beam1 = rsrp_sery_widebeam_m(j, k, 16);
                else
                    neibor_beam1 = rsrp_sery_widebeam_m(j, k, max_location - 1);
                end
                if(max_location == 16)
                    neibor_beam2 = rsrp_sery_widebeam_m(j, k, 1);
                else
                    neibor_beam2 = rsrp_sery_widebeam_m(j, k, max_location + 1);
                end
                if(neibor_beam1 > neibor_beam2)
                    if(neibor_beam1 > (1 / th(1, 2, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 1;
                        power_ratio_v(count, k) = power_ratio_v(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                    else
                        optimal = (max_location - 1) * n_beam / w_beam + 2;
                        power_ratio_v(count, k) = power_ratio_v(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                    end
                else
                    if(neibor_beam2 > (1 / th(2, 6, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 4;
                        power_ratio_v(count, k) = power_ratio_v(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                    else
                        optimal = (max_location - 1) * n_beam / w_beam + 3;
                        power_ratio_v(count, k) = power_ratio_v(count, k, 1) + rsrp_sery_no_noise_m(j, k, optimal)^2;
                    end
                end
            end
        end
    end
end
power_ratio_v = power_ratio_v / 256 / 32;

save('baseline1_evaluation.mat', 'power_ratio', 'pdf', 'power_ratio_v');