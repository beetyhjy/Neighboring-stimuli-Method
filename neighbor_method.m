%% Clear workspace
clear
close all
clc

%% Fixed parameter (Modify according to the experimental setting)
% Sampling rate [Hz]
fs = 250;                  

% Duration for gaze shifting [s]
len_shift_s = 0.5;                  

% List of stimulus frequencies
list_freqs = [8:1:15 8.2:1:15.2 8.4:1:15.4 8.6:1:15.6 8.8:1:15.8]; % benchmark datatset

num_cols = 8;
num_rows = 5;

% The number of stimuli
num_targets = length(list_freqs);    

% Labels of data
labels = [1:1:num_targets];  

%% Parameter for analysis (Modify according to your analysis)
% Visual latency being considered in the analysis [s]
len_delay_s = 0.14 + 0.5;            

% Visual latency [samples]
len_delay_smpl = round(len_delay_s*fs);    

% The number of sub-bands in filter bank analysis
num_fbs = 5;

fb_coefs = [1:num_fbs].^(-1.25)+0.25;

samples = 5*fs;

time_windows = 0.2:0.2:1; 

channel_selection = [48 53 55 56 57 59 61 62 63]; % 9 channels
%% (1) Defining the neighboring stimuli corresponding to the target stimulus
neighbors = cell(num_targets,1);
for targ_i = 1:1:num_targets
    potential_stimuli = [targ_i-num_cols,targ_i-1,targ_i,...
                         targ_i+1,targ_i+num_cols];
    stimuli = intersect(potential_stimuli,labels);
%     num_stimuli = length(stimuli);
    neighbors(targ_i,:) = {stimuli};
end
%% (2) Performing the SSVEP detection algorithm
% Preparing data
for subject = 1:35    
    filename =[ '~\Benchmark dataset\S' num2str(subject) '.mat'];
    load(filename); % data 64 channels * 1500 points * 40 targets * 6 trials(blocks)
    
    eeg = data(channel_selection,len_delay_smpl+1:len_delay_smpl+samples,:,:);
    [num_chans,~, ~, num_blocks] = size(eeg);
    fprintf('Subject %d:\n',subject);
    
    for loocv_i =  1:1:num_blocks
        testtrial = loocv_i;
%% (2-1) Training stage        
        train_all = eeg;
        train_all(:, :, :, testtrial) = [];
        num_trials = size(train_all,4);

        trains = zeros(num_targets,num_fbs,num_chans,samples);
        W = zeros(num_fbs,num_targets,num_chans); 

        for targ_i = 1:num_targets
            template = squeeze(mean(train_all(:,:,targ_i,:),4)); 

            stimuli = cell2mat(neighbors(targ_i));    
            num_stimuli = length(stimuli);

            trains_data = squeeze(train_all(:,:,stimuli,:));

            for fb_i = 1:num_fbs 
                fb_tmp = filterbank(template, fs, fb_i); % filter bank analysis 
                trains(targ_i, fb_i, :, :) = fb_tmp;

                autocov_data = zeros(num_stimuli,num_chans,num_chans);
                cov_data = zeros(num_stimuli,num_chans,num_chans);
                for stimulus_i = 1:num_stimuli
                    stimulus_data = squeeze(trains_data(:,:,stimulus_i,:));
                    fb_data = filterbank(stimulus_data, fs, fb_i); % filter bank analysis 

                    UX = reshape(fb_data, num_chans, samples*num_trials);
                    UX = bsxfun(@minus, UX, mean(UX,2));
                    autocov_data(stimulus_i,:,:) = UX*UX'/(samples*num_trials);  
                    
                    data_avg = squeeze(mean(fb_data,3));
                    data_avg = bsxfun(@minus,data_avg,mean(data_avg,2));
                    cov_data(stimulus_i,:,:) = (data_avg * data_avg')/samples;
                end
    
                S = squeeze(sum(cov_data,1));
                Q = squeeze(sum(autocov_data,1));
    
                [evecs,evals] = eig(S,Q);
                [~,comp2plot] = max(diag(evals)); % find maximum component
                evecs = bsxfun(@rdivide,evecs,sqrt(sum(evecs.^2,1)));
                w = evecs(:,comp2plot);            
                W(fb_i, targ_i, :) = w;  
            end % fb_i           
        end       
        
%% (2-2) Test Stage
        for tw_i = 1:length(time_windows)
            len_gaze_s = time_windows(tw_i);
            len_sel_s = len_gaze_s+len_shift_s;
            len_gaze_smpl = round(len_gaze_s*fs);
            for targ_i = 1:num_targets                
                test = eeg(:,1:len_gaze_smpl,targ_i,testtrial); 
                
                r = zeros(num_fbs,num_targets);
                for fb_i = 1:num_fbs
                    testdata = filterbank(test, fs, fb_i);                   
                    for targs_j = 1:num_targets
                        tmp = squeeze(trains(targs_j,fb_i,:,1:len_gaze_smpl));
                        w = squeeze(W(fb_i, targs_j, :));
%                             w = squeeze(W(fb_i, :, :))'; % ensemble
                        r_tmp = corrcoef(w'*testdata,w'*tmp);
                        r(fb_i,targs_j) = r_tmp(1,2);
                    end % targs_j
                end % fb_i
                rho = fb_coefs*r;                
                [~,estimated(targ_i)] = max(rho); 
%                     Rho(subject,loocv_i,targs_i,:) = real(rho);                
            end
           
            % Evaluation
            estmt_cm = confusionmat(labels,estimated);
            true_cm = diag(ones(1,num_targets));
            all = numel(estmt_cm);
            tp = sum(sum((estmt_cm == 1) & (true_cm == 1)));
            fp = sum(sum((estmt_cm == 0) & (true_cm == 1)));
            fn = sum(sum((estmt_cm == 1) & (true_cm == 0)));
            tn = sum(sum((estmt_cm == 0) & (true_cm == 0)));

            acc = tp/num_targets*100
            Acc(tw_i,testtrial) = acc;

        end
    end
    Accuracies(subject,:,:) = Acc;
end

