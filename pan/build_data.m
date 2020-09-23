clear;
close all;
clc;
addpath /home/daniele/Research/MPanSuite
MPanSuiteInit;

%%

%%% OU parameters
alpha = 0.5;
mu = 0;
c = 0.5;

%%% simulation parameters
frand = 10;      % [Hz] sampling rate of the random signal
tstop = 5 * 60;  % [s] 5 minutes

%%% inertia values
H_min = 2;
H_max = 10;
dH = 1;
H = H_min : dH : H_max;
N_H = length(H);

%%% how many trials per inertia value
N_trials = 1000;
seeds = randi(1000000, [N_H, N_trials]);

dt = 1 / frand;
t = dt + (0 : dt : tstop)';
N_samples = length(t);
load_noise = zeros(N_H, N_trials, N_samples);

coeff_1 = alpha * mu * dt;
coeff_2 = 1 / (1 + alpha * dt);
noise = c * sqrt(dt) * randn(size(load_noise));

for i = 1 : N_H
    for j = 1 : N_trials
        for k = 1 : N_samples - 1
            load_noise(i, j, k+1) = (load_noise(i, j, k) + ...
                coeff_1 + noise(i, j, k)) * coeff_2;
        end
    end
end

%%
pan_file = 'ieee14';
MPanLoadNet([pan_file, '.pan']);

MPanAlter('Altstop', 'TSTOP', tstop);
MPanAlter('Alfrand', 'FRAND', frand);

names = {'G1', 'G2', 'G3', 'G6', 'G8', 'coi'};
var_names = {'omega01', 'omega02', 'G3:omega', 'G6:omega', 'G8:omega', 'omegacoi'};

G1_alter_opt = MPanAlterSetOptions('instance', 'G1', ...
                                   'invalidate', false, ...
                                   'annotate', 1);

for i = 1 : N_H

    MPanAlter(sprintf('Al%d', i), 'm', 2 * H(i), G1_alter_opt);
    
    for name = names
        eval(['omega_', name{1}, ' = zeros(N_trials, N_samples-1);']);
    end
    
    for j = 1 : N_trials
        noise_samples = [t, squeeze(load_noise(i, j, :))];
        
        tran_name = sprintf('Tr_%d_%d', i, j);
        cmd = sprintf(['Tr_%d_%d tran stop=TSTOP nettype=1 method=2 maxord=2 ', ...
            'noisefmax=FRAND/2 noiseinj=2 seed=%d iabstol=1u devvars=1 ', ...
            'tmax=0.1 savelist=SaveSet mem=["time:noise", "omega01:noise", ', ...
            '"omega02:noise", "G3:omega:noise", "G6:omega:noise", ', ...
            '"G8:omega:noise", "omegacoi:noise"] annotate=0'], i, j, seeds(i, j));
        pansimc(cmd);

        for k = 1 : length(names)
            cmd = sprintf('omega_%s(j,:) = panget(''%s.%s:noise'');', ...
                names{k}, tran_name, var_names{k});
            eval(cmd);
        end
    end

    out_file = sprintf('ieee14_training_set_H=%g.mat', H(i));

    time = panget([tran_name, '.time:noise'])';
    load_seeds = seeds(i, :);
    inertia = H(i);
    noise = squeeze(load_noise(i, :, 1:end-1));
    save(out_file, 'time', 'omega_G1', 'omega_G2', 'omega_G3', 'omega_G6', ...
        'omega_G8', 'omega_coi', 'load_seeds', 'inertia', 'noise');
end

