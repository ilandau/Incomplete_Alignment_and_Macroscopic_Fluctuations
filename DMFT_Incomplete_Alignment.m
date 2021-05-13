% Must define V_hat, f, and sigma


ac_time_dmft=25;
dt_dmft=.2;


% construct a 2d lattice for integrating over Gaussian variables
nbins =800;
w = linspace(-5.5,5.5,nbins);
[Z,X] = meshgrid(w,w);

options = optimoptions('fsolve','Display','off');

phi=@(x) tanh(x);                   % transfer Fn
dphi = @(x) 1- tanh(x).^2;          % 1st deriv of transfer Fn
Phi = @(x) log(cosh(x));            % integral of transfer Fn - for 'Potential Energy Fn'
%% Fns for Balance Subspace Time-Avg


% Fn: returns norm of R_hat given the norm of H_hat and Delta_0 (variance in orthogonal subspace)
r_hat_norm_fn = @(h,d) h.*trapz(w, normpdf(w).*dphi(sqrt(d+h.^2)*w));
    % Note: In Gaussian Structured Connectivity setting, norm of R_hat is proportonal to norm of H_hat

% Fn: given Delta_0, solve for norm of H_hat, by requiring balance constraint
h_hat_norm_solve_fn = @(d) abs(fsolve(@(hn) r_hat_norm_fn(hn,d) - norm(R_hat_Theory),1,options));


%% Fns for DMFT in Orthogonal Subspace

% Fn: Autocorrelation of single units, given Delta_0, Delta(tau), and norm of H_hat
AC_R_fn = @(d,d0,h) trapz(w, normpdf(w).*trapz(w, normpdf(X).*phi(sqrt(d0-abs(d))*X + sign(d)*sqrt(abs(d)+h^2)*Z)).^2);

% Fn: Potential Energy at tau=0, given Delta_0 and norm of H_hat
V0 = @(d0,h) -d0.^2/2+g^2*trapz(w, normpdf(w).*Phi(sqrt(d0+h^2)*w).^2);
% Fn: Potential Energy at tau=infty, given Delta_0, Delta_inf, and norm of H_hat
Vinf = @(d_inf,d0,h) -d_inf.^2/2+g^2*trapz(w, normpdf(w).*trapz(w, normpdf(X).*Phi(sqrt(d0-d_inf)*X+sqrt(h^2+d_inf)*Z)).^2);


% Fn: Solve for Delta_inf by requiring Delta_inf = g^2 * var(R_time_avg)
Dinf_Fn = @(d0,hn) fsolve(@(d_inf) d_inf - g^2*AC_R_fn(d_inf,d0,hn),0.5,optimoptions(@fsolve,'Display','off'));
    % Note: Var of R_time_avg is given by Autocorrelation at infinite time lag

% Fn: Solve for Delta_0 by requiring potential energy is constant btw tau=0 and tau=inf
Delta0_fn = @(hn) fsolve(@(d0) V0(d0,hn)-Vinf(Dinf_Fn(d0,hn),d0,hn),6,optimoptions(@fsolve,'TolFun',1e-16,'Display','off'));

% Solve for Delta_0 - total variance of H_perp
Delta0_Th = real(fsolve(@(d) Delta0_fn(h_hat_norm_solve_fn(d))-d,rand()+g,options));

% Now solve for norm of H_hat
H_hat_Norm_Th = h_hat_norm_solve_fn(Delta0_Th);

% Find Delta_inf
Delta_inf_Th = Dinf_Fn(Delta0_Th,H_hat_Norm_Th);

Delta_temporal_Th = Delta0_Th - Delta_inf_Th;

% Find total variance of activity
C0_Th = AC_R_fn(Delta0_Th,Delta0_Th,H_hat_Norm_Th);



%% Compute full autocorrelation Fn

% 2nd Order Differential Eqn for Delta
F = @(t,Delta) [0,1;1,0]*Delta-[0;g^2*AC_R_fn(Delta(1),Delta0_Th,H_hat_Norm_Th)];
    % Note: 1st variable is Delta(tau), 1nd variable is its time derivative
    % Initial condition is therefore [Delta0_Th; 0]

opts = odeset('RelTol',1e-4,'AbsTol',1e-6);
tic
[t_sol,Del_2d_sol] = ode45(F,0:dt_dmft:ac_time_dmft, [Delta0_Th;0],opts);
toc

Delta_tau_Th = Del_2d_sol(:,1);


C_tau_Th = arrayfun(@(d) AC_R_fn(d,Delta0_Th,H_hat_Norm_Th),Delta_tau_Th);

C_inf_Th = C_tau_Th(end);

% avg single-neuron mean-subtracted autocorrelation.
C_temporal_Th = C0_Th - C_inf_Th;

