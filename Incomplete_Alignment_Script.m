% All time is measured in units of the neuron's intrinsic time-scale

%% Connectivity Parameters 
compute_DMFT = true;

N = 4000;   % Number of neurons
D = 3;      % Number of modes in Low-Rank Structure
g=2.5;      % Strength of Random Connectivity


R_hat_Norm = 0.2; % set desired norm of balance firing rates
f_hat_Norm = 0.2; % OR set norm of external drive
set_R_hat_Norm = true;  % if 'true', rescale f_hat in order to generate desired balance firing rate norm
                        % if 'false', use given f_hat_Norm

                        
sigmas_M = linspace(0.8,1.2,D)';  % set singular values of Structured Connectivity, M
alignment_determinant = 0.7;    % set determinant of Alignment Matrix (V_hat)


f_hat = linspace(-2,6,D)';      % set external drive to balance subspace
f_hat = f_hat_Norm*f_hat/norm(f_hat);


%%

% Non-linearity and its derivative
phi = @(x) tanh(x);
dphi = @(x) 1- tanh(x).^2;


%seed_vhat = randi(1000);                  % seed for Alignment Matrix
seed_U =randi(100);                      % seed for structure of the Balance Subspace, U
seed_random_connectivity = randi(100);   % seed for random component of connectivity
seed_initial_cond_dynamics = randi(100); % seed for initial condition of dynamics


%% Simulation time parameters

dt = 0.1;      % Time step for output of solver

ac_time = 50; % Time lag for computing autocorrelation (in units of intrinsic time-scale)
ac_lags = ac_time/dt; % Number of lags for computing autocorrelation


t_record = 200; % Time to record
t_buffer= 50;   % Buffer time to run before recording

 
%% Construct Alignment Matrix (V_hat)

% Set singular values of Alignment Matrix so that their product gives desired determinant
log_avg = 1/D*log(alignment_determinant);
logsigs = linspace(2*log_avg,0,D);
alignment_singular_vals = exp(logsigs);

         
rng(seed_vhat);      % Initialize seed of random number generator for Alignment Matrix

tries=0;
tryagain=true;

% Loop until all eigs have negative real  part
while tryagain
    
    if tries > 200
        error('Stuck trying to build Alignment Matrix. Try smaller D')
    end
    
    QL = normrnd(0,1,D,D); % left singular-vectors of Alignment Matrix
    [QL,~] = qr(QL);
    QR = normrnd(0,1,D,D); % right singular-vectors of Alignment Matrix
    [QR,~] = qr(QR);

    V_hat = QL*diag(alignment_singular_vals)*QR';
    
    alignment_eigs= eig(V_hat);
    tries=tries+1;
    if prod(real(alignment_eigs) < 0)    % if all eigs have neg real part, we're done
        tryagain=false;
    elseif prod(real(alignment_eigs)>0)  % if all eigs are positive, just flip sign of QR
        QR = -QR;
        V_hat = -V_hat;
        tryagain=false;   
    end                                 % else try again
end

%% Compute R_hat_Theory and rescale f_hat if desired

% Set R_hat_Theory to solve balance equations
R_hat_Theory = - QL*(diag(1./alignment_singular_vals))*QR'*(f_hat./sigmas_M);

% If desired, rescale f_hat to produce desired norm of R_hat
if set_R_hat_Norm
    rescale_factor = R_hat_Norm/norm(R_hat_Theory);
    f_hat = f_hat*rescale_factor;  % This sets norm(uRTh) == R_hat_Norm
    
    R_hat_Theory = - QL*(diag(1./alignment_singular_vals))*QR'*(f_hat./sigmas_M);
end

if compute_DMFT
    DMFT_Incomplete_Alignment
    
    % Balance Subspace Covariance Matrix (at tau=0)
    C0_hat_mat_Th = C_temporal_Th*QL*(diag(1./alignment_singular_vals.^2)-eye(D))*QL';
end

%% Construct Structured Connectivity - M
  
rng(seed_U) % Initialize random number generator for structure of Balance Subspace, U

O = normrnd(0,1,N,N);
[U_norm1,~] = qr(O);                    % Generate random orthogonal vectors with norm 1

U = sqrt(N)*U_norm1(:,1:D);             
U_perp = sqrt(N)*U_norm1(:,D+(1:D)); 

V_perp = U_perp*sqrt(eye(D)-diag(alignment_singular_vals).^2)*QR';
V = U*V_hat + V_perp;

M = 1/sqrt(N)*U*diag(sigmas_M)*V';  % Set structured component of connectivity

%% Construct All Connectivity

rng(seed_random_connectivity);  % Initialize seed for random component of connectivity
J = normrnd(0,g/sqrt(N),N,N);   % Set random component of connectivity
W = M + J;                      % Set total recurrent connectivity

f = sqrt(N)*U*f_hat;            % Set external drive in standard neuron basis

%% Run Simulation

if t_buffer==0
    t_buffer = dt;
end

t_end = t_buffer+t_record; 
Ts = [0,t_buffer:dt:t_end];  % Vector of times to output from solver
  
rng(seed_initial_cond_dynamics) % Initialize seed of random number generator for initial condition

opts = odeset('RelTol',1e-3,'AbsTol',1e-5);
H0 = randn(N,1);% Initial condition
Func = @(t,h) (-h+W*phi(h)+f*(1)); % Define dh/dt for differential eqn solver, ode4

tic
[t,Hsol] = ode45(Func,Ts,H0,opts); % Run simulation
toc

tt=t(2:end)-t_buffer;   % time steps corresponding to output
T = length(tt);         % Number of time steps

H = Hsol(2:end,:)';     % Rearrange recorded dynamics so that each row is a neuron, with recording from t_start until t_end in time steps of dt

%%

R = phi(H);             % Neural Activity (N x T)

H_hat = U'*H/N;        % Balance Subspace of input current (D x T)
R_hat = U'*R/N;        % Balance Subspace Activity (D x T)

R_TimeAvg = mean(R,2);          % Mean single-neuron activity (N x 1)
R_hat_TimeAvg = U'*R_TimeAvg/N; % Mean balance subspace activity (D x 1)


H_hat_TimeAvg = mean(H_hat,2); % Mean balance subspace input dynamics (N x 1)

H_perp = H-U*H_hat;                 % Dynamics in orthogonal subspace (N x T)
H_perp_TimeAvg = mean(H_perp,2);    % Single-neuron time-avg in orthogonal subspace (N x 1)

Delta0_Sim = var(H_perp(:));       % Total variance in orthogonal subspace (Delta_0)
Delta_inf_Sim = var(H_perp_TimeAvg); % Var of single-neuron time-avg (Delta_inf)
Delta_temporal_Sim = mean(var(H_perp,1,2)); % Temporal Var of single neurons (Delta_0 - Delta_inf)

C_tau_two_sided_Sim = AvgAutocorr(R,ac_lags);
C_tau_Sim = C_tau_two_sided_Sim(ac_lags+1:end);


% ACperpH = AvgAutocorr(H_perp,ac_lags);
%%

% Plot activity trace of balance subsapce and 5 random neurons

figure; plot(tt, R(randi(N,5,1),:))
hold on; plot(tt, R_hat,'-k','LineWidth',2)
plot(tt, R_hat_Theory*ones(1,T),'--k')
title(sprintf('Sample Activity Traces, N=%d, D=%d, g=%.1f',N,D,g))
xlabel('time')


%%

C0_hat_mat_Sim = cov(R_hat')*N;
mx = max([C0_hat_mat_Th(:);C0_hat_mat_Sim(:)]);
mn = min([C0_hat_mat_Th(:);C0_hat_mat_Sim(:)]);

figure;
subplot(121)
imagesc(C0_hat_mat_Sim)
colorbar
set(gca,'XTick',1:D,'YTick',1:D)
xlabel('Structured Subsapce Modes')
ylabel('Structured Subsapce Modes')
colormap jet
caxis([mn,mx])
title(sprintf('C_{hat}*N -- Simulation, N = %d',N))
subplot(122)
imagesc(C0_hat_mat_Th)
colorbar
colormap jet
caxis([mn,mx])
title('Theory')
set(gca,'XTick',1:D,'YTick',1:D)
xlabel('Balance Subsapce Modes')
ylabel('Balance Subsapce Modes')


%% Plot autocorrelation C(tau) (avg AC of single-neurons)

figure;plot((0:ac_lags)*dt,C_tau_Sim)
hold on;plot(t_sol,C_tau_Th,'--')
title(sprintf('Avg single-neuron autocorrelation, N=%d',N))
legend('Simulation','Theory')
