clear;
close all;
clc;
addpath /home/daniele/Research/MPanSuite
MPanSuiteInit;
rng('default');

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
N_trials = 100;
random_seeds = randi(1000000, [N_H, N_trials]);
% fid = fopen('/dev/random', 'rb');
% random_seeds = mod( fread(fid, [N_H, N_trials], 'uint32'), 1000000 );
% fclose(fid);

dt = 1 / frand;
t = dt + (0 : dt : tstop)';
N_samples = length(t);
coeff_1 = alpha * mu * dt;
coeff_2 = 1 / (1 + alpha * dt);

suffix = 'train';
if strcmp(suffix, 'test')
    H = H + 1/3;
elseif strcmp(suffix, 'validation')
    H = H + 2/3;
end

%%
pan_file = 'ieee14_valid';
MPanLoadNet([pan_file, '.pan']);

alter_opt = MPanAlterSetOptions('annotate', 1);
MPanAlter('Altstop', 'TSTOP', tstop, alter_opt);
MPanAlter('Alfrand', 'FRAND', frand, alter_opt);

names = {'G1', 'G2', 'G3', 'G6', 'G8', 'coi'};
var_names = {'omega01', 'omega02', 'G3:omega', 'G6:omega', 'G8:omega', 'omegacoi'};

G1_alter_opt = MPanAlterSetOptions(alter_opt, 'instance', 'G1', ...
                                   'invalidate', false);

for name = names
    eval(['omega_', name{1}, ' = zeros(N_trials, N_samples-1);']);
end

noise = zeros(N_trials, N_samples);

for i = 1 : N_H

    MPanAlter(sprintf('Al%d', i), 'm', 2 * H(i), G1_alter_opt);
    
    for j = 1 : N_trials

        rng(random_seeds(i, j));
        rnd = c * sqrt(dt) * randn([1, N_samples]);

        for k = 1 : N_samples - 1
            noise(j, k+1) = (noise(j, k) + ...
                coeff_1 + rnd(k)) * coeff_2;
        end
        noise_samples = [t, noise(j, :)'];
        
        tran_name = sprintf('Tr_%d_%d', i, j);
        cmd = sprintf(['Tr_%d_%d tran stop=TSTOP nettype=1 method=2 maxord=2 ', ...
            'noisefmax=FRAND/2 noiseinj=2 seed=%d iabstol=1u devvars=1 ', ...
            'tmax=0.1 savelist=SaveSet mem=["time:noise", "omega01:noise", ', ...
            '"omega02:noise", "G3:omega:noise", "G6:omega:noise", ', ...
            '"G8:omega:noise", "omegacoi:noise"] annotate=0'], i, j, random_seeds(i, j));
        pansimc(cmd);

        for k = 1 : length(names)
            cmd = sprintf('omega_%s(j,:) = panget(''%s.%s:noise'');', ...
                names{k}, tran_name, var_names{k});
            eval(cmd);
        end
    end

    time = panget([tran_name, '.time:noise'])';
    seeds = random_seeds(i, :);
    inertia = H(i);

    out_file = sprintf('ieee14_%s_set_H_%.3f.mat', suffix, H(i));
    save(out_file, 'time', 'omega_G1', 'omega_G2', 'omega_G3', 'omega_G6', ...
        'omega_G8', 'omega_coi', 'seeds', 'inertia', 'noise');
end

