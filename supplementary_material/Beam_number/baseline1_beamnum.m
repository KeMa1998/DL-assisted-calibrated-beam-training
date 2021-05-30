clear all;
close all;
clc;

%Baseline 1
%VTC 2019, Xingyi Luo, Wendong Liu et al.
%Calibrated Beam Training for Millimeter-Wave Massive MIMO Systems

%% 32 beams
sector_start=-pi/2; % Sectoar start point
sector_end=pi/2; % Sector end point
%establish look-up table
% narrow beam number
n_beam = 32;
% wide beam number
w_beam = 16;
% wide beam antenna number
rx = 16;

% save normalized beamforming gain
power_ratio = zeros(10, 1);
count = 0;
for P = [15]
    for i = 65 : 80
        load(['..\dataset\testing_15dBm_32beam\data_TCOM_16Tx_64Tx_32beam_RK8dB_testing_' num2str(i) '_' num2str(P) 'dBm.mat']);
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
                    optimal = (max_location - 1) * n_beam / w_beam + 1;
                else
                    optimal = (max_location - 1) * n_beam / w_beam + 2;
                end
                power_ratio(k) = power_ratio(k) + rsrp_sery_no_noise_m(j, k, optimal)^2;
            end
        end
    end
end
power_ratio = power_ratio / 256 / 16;

%save('baseline1_evaluation_32beam.mat', 'power_ratio');


%% 128 beams
sector_start=-pi/2; % Sectoar start point
sector_end=pi/2; % Sector end point
%establish look-up table
% narrow beam number
n_beam = 128;
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

% save normalized beamforming gain
power_ratio = zeros(10, 1);
count = 0;
for P = [15]
    for i = 65 : 80
        load(['..\dataset\testing_15dBm_128beam\data_TCOM_16Tx_64Tx_128beam_RK8dB_testing_' num2str(i) '_' num2str(P) 'dBm.mat']);
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
                    if(neibor_beam1 > (1 / th(1, 4, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 1;
                    elseif(neibor_beam1 > (1 / th(1, 8, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 2;
                    elseif(neibor_beam1 > (1 / th(1, 12, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 3;
                    else
                        optimal = (max_location - 1) * n_beam / w_beam + 4;
                    end
                else
                    if(neibor_beam2 > (1 / th(2, 12, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 8;
                    elseif(neibor_beam2 > (1 / th(2, 8, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 7;
                    elseif(neibor_beam2 > (1 / th(2, 4, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 6;
                    else
                        optimal = (max_location - 1) * n_beam / w_beam + 5;
                    end
                end
                power_ratio(k) = power_ratio(k) + rsrp_sery_no_noise_m(j, k, optimal)^2;
            end
        end
    end
end
power_ratio = power_ratio / 256 / 16;

%save('baseline1_evaluation_128beam.mat', 'power_ratio');


%% 256 beams
sector_start=-pi/2; % Sectoar start point
sector_end=pi/2; % Sector end point
%establish look-up table
% narrow beam number
n_beam = 256;
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

% save normalized beamforming gain
power_ratio = zeros(10, 1);
count = 0;
for P = [15]
    for i = 65 : 80
        load(['..\dataset\testing_15dBm_256beam\data_TCOM_16Tx_64Tx_256beam_RK8dB_testing_' num2str(i) '_' num2str(P) 'dBm.mat']);
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
                    if(neibor_beam1 > (1 / th(1, 4, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 8;
                    elseif(neibor_beam1 > (1 / th(1, 8, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 7;
                    elseif(neibor_beam1 > (1 / th(1, 12, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 6;
                    elseif(neibor_beam1 > (1 / th(1, 16, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 5;
                    elseif(neibor_beam1 > (1 / th(1, 20, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 4;
                    elseif(neibor_beam1 > (1 / th(1, 24, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 3;
                    elseif(neibor_beam1 > (1 / th(1, 28, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 2;
                    else
                        optimal = (max_location - 1) * n_beam / w_beam + 1;
                    end
                else
                    if(neibor_beam2 > (1 / th(2, 28, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 16;
                    elseif(neibor_beam2 > (1 / th(2, 24, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 15;
                    elseif(neibor_beam2 > (1 / th(2, 20, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 14;
                    elseif(neibor_beam2 > (1 / th(2, 16, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 13;
                    elseif(neibor_beam2 > (1 / th(2, 12, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 12;
                    elseif(neibor_beam2 > (1 / th(2, 8, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 11;
                    elseif(neibor_beam2 > (1 / th(2, 4, max_location)))
                        optimal = (max_location - 1) * n_beam / w_beam + 10;
                    else
                        optimal = (max_location - 1) * n_beam / w_beam + 9;
                    end
                end
                power_ratio(k) = power_ratio(k) + rsrp_sery_no_noise_m(j, k, optimal)^2;
            end
        end
    end
end
power_ratio = power_ratio / 256 / 16;

%save('baseline1_evaluation_256beam.mat', 'power_ratio');