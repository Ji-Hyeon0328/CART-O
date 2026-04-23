%% main_call_impedance_v2_demo.m
% Full prototype pipeline with Version-2 impedance residual.
%
% Version-2 residual:
%   Kp_eff = Kp_nom - gate * beta_imp  * tanh(s_imp/s0) * I
%   Kd_eff = Kd_nom + gate * alpha_imp * tanh(s_imp/s0) * I
%
% Compared with Version 1, Version 2 also lowers stiffness near touchdown
% when impact is detected.

clear; clc;

%% Example inputs
z_t = 0;  % 0 = conservative, 1 = aggressive

a_HL = [0.20; -0.30; 0.40; -0.10];   % [a_swing, a_body, a_duty, a_imp]'
beta_t = [0.30; 0.40; 0.30];         % [beta_h, beta_v, beta_e]'

x_hat = [
    0.00; 0.00; 0.42; ...      % p = [x,y,z]
    0.03; -0.02; 0.10; ...     % eta = [roll,pitch,yaw]
    0.25; 0.02; 0.00; ...      % v = [vx,vy,vz]
    0.00; 0.00; 0.05           % omega = [wx,wy,wz]
];

u_cmd = [0.30; 0.00; 0.10];

%% Common parameters
params.dt = 0.02;
params.N  = 20;
params.robot.mass = 25.0;
params.robot.g    = 9.81;
params.robot.Ibody = diag([0.45, 1.20, 1.30]);

% Leg order: [LF, RF, LH, RH]
params.hip_offset_body = [
     0.28,  0.28, -0.28, -0.28;
     0.16, -0.16,  0.16, -0.16;
     0.00,  0.00,  0.00,  0.00
];

params.p_foot_now = [
     0.30,  0.30, -0.28, -0.28;
     0.18, -0.18,  0.18, -0.18;
     0.00,  0.00,  0.00,  0.00
];

params.Kv_foot = diag([-0.08, -0.08, 0.0]);

%% Force MPC parameters
mpcParams = struct();
mpcParams.dt = params.dt;
mpcParams.N  = params.N;
mpcParams.robot = params.robot;
mpcParams.fz_max_per_leg = 1.2 * params.robot.mass * params.robot.g;
mpcParams.use_force_rate_bound = false;
mpcParams.df_max = 500.0;
mpcParams.rho_f  = 0.60;
mpcParams.rho_df = 0.40;

mpcParams.q0.x  = 0.0;  mpcParams.q0.y  = 0.0;  mpcParams.q0.z  = 80.0;
mpcParams.q0.roll = 80.0; mpcParams.q0.pitch = 80.0; mpcParams.q0.yaw = 1.0;
mpcParams.q0.vx = 40.0; mpcParams.q0.vy = 40.0; mpcParams.q0.vz = 30.0;
mpcParams.q0.wx = 1.0; mpcParams.q0.wy = 1.0; mpcParams.q0.wz = 30.0;
mpcParams.weightRange.wh = [0.5, 3.0];
mpcParams.weightRange.wv = [0.5, 3.0];
mpcParams.weightRange.we = [0.1, 5.0];
mpcParams.H_reg = 1e-8;

%% WBC parameters
wbcParams = struct();
wbcParams.dt = params.dt;
wbcParams.robot = params.robot;
wbcParams.fz_max_per_leg = mpcParams.fz_max_per_leg;
wbcParams.mu = [];  % if empty, use Theta.ctrl.mu_exp
wbcParams.tau_min = -90 * ones(12,1);
wbcParams.tau_max =  90 * ones(12,1);
wbcParams.tau_prev = zeros(12,1);
wbcParams.Kp_base = diag([40, 40, 80, 80, 80, 30]);
wbcParams.Kd_base = diag([10, 10, 20, 20, 20, 8]);
wbcParams.Kp_foot = diag([80, 80, 120]);
wbcParams.Kd_foot = diag([10, 10, 15]);
wbcParams.Wb = diag([20, 20, 100, 100, 100, 20]);
wbcParams.Wfoot = diag([50, 50, 100]);
wbcParams.Wforce_per_foot = diag([1, 1, 3]);
wbcParams.Wtau = 1e-3 * eye(12);
wbcParams.Wdtau = 1e-2 * eye(12);
wbcParams.H_reg = 1e-8;

%% Impedance residual parameters, Version 2
impParams = struct();
impParams.lambda_res = 0.20;
impParams.phase_width = 0.10;
impParams.touchdown_phase = 1.0;

% Version-2 impact adaptation
impParams.alpha_imp = 25.0;   % damping increase scale
impParams.beta_imp  = 20.0;   % stiffness decrease scale
impParams.s_imp0    = 0.25;

% Nominal impedance lookup tables
impParams.Kp.cons.st = diag([25, 25, 45]);
impParams.Kd.cons.st = diag([18, 18, 30]);
impParams.Kp.cons.sw = diag([60, 60, 90]);
impParams.Kd.cons.sw = diag([ 7,  7, 10]);
impParams.Kp.agg.st = diag([40, 40, 70]);
impParams.Kd.agg.st = diag([20, 20, 35]);
impParams.Kp.agg.sw = diag([90, 90, 130]);
impParams.Kd.agg.sw = diag([10, 10, 15]);

impParams.Kp_min = diag([5, 5, 5]);
impParams.Kp_max = diag([200, 200, 250]);
impParams.Kd_min = diag([1, 1, 1]);
impParams.Kd_max = diag([80, 80, 100]);

% Mock foot state for demo visibility.
impParams.mock_position_error = [
    0.015, -0.010,  0.000,  0.005;
   -0.005,  0.000,  0.010, -0.010;
    0.000,  0.000, -0.005, -0.005
];

impParams.mock_foot_velocity_now = [
    0.02,  0.00,  0.01, -0.01;
    0.00,  0.01, -0.01,  0.00;
   -0.05, -0.03, -0.20, -0.10
];

%% Run pipeline
Theta = thetaDecoder(z_t, a_HL, x_hat, u_cmd);
Ref   = thetaRefMapper(Theta, x_hat, u_cmd, params);
MPC   = forceMPC(x_hat, Ref, Theta, beta_t, mpcParams);

qj  = zeros(12,1);
dqj = zeros(12,1);
WBC = wbcQP_mock(x_hat, qj, dqj, Ref, Theta, MPC, wbcParams);
IMP = impedanceResidualV2(z_t, Theta, Ref, WBC, impParams);

%% Display outputs
disp("===== WBC tau_OC ====="); disp(WBC.tau_OC);
disp("===== Version-2 impedance residual tau_res ====="); disp(IMP.tau_res);
disp("===== Final torque tau_final ====="); disp(IMP.tau_final);
disp("===== Per-leg gain adaptation summary =====");
disp(table((1:4)', IMP.impact_signal, IMP.phase_gate, IMP.Kp_delta_diag, IMP.Kd_delta_diag, ...
    'VariableNames', {'Leg','s_imp','phase_gate','mean_dKp','mean_dKd'}));

%% Visualization
figure;
imagesc(Ref.S);
colormap(gray);
xlabel("Horizon step k"); ylabel("Leg index: 1 LF, 2 RF, 3 LH, 4 RH");
title("Contact schedule S_t, white=stance, black=swing"); colorbar;

figure;
plot(0:params.N-1, MPC.F_by_leg_z');
xlabel("Horizon step k"); ylabel("Vertical GRF f_z [N]");
legend("LF","RF","LH","RH"); title("Optimized vertical GRF over horizon"); grid on;

figure;
plot(0:params.N-1, sum(MPC.F_by_leg_z,1), "LineWidth", 1.5); hold on;
yline(params.robot.mass * params.robot.g, "--");
xlabel("Horizon step k"); ylabel("Total vertical GRF [N]");
legend("sum f_z", "mg"); title("Total vertical GRF vs body weight"); grid on;

figure;
bar([WBC.tau_OC, IMP.tau_res, IMP.tau_final]);
xlabel("Joint index"); ylabel("Torque [Nm]");
legend("tau OC", "tau residual", "tau final");
title("WBC torque + Version-2 impedance residual"); grid on;

figure;
bar(IMP.impact_signal);
xlabel("Leg index"); ylabel("s imp"); title("Impact signal per leg"); grid on;

figure;
bar(IMP.phase_gate);
xlabel("Leg index"); ylabel("gate"); title("Touchdown phase gate per leg"); grid on;

figure;
bar([IMP.Kp_delta_diag, IMP.Kd_delta_diag]);
xlabel("Leg index"); ylabel("Mean diagonal gain delta");
legend("mean diag Delta Kp", "mean diag Delta Kd");
title("Version-2 gain adaptation summary"); grid on;
