%% main_call_impedance_v3_demo.m
% Full prototype pipeline:
%   Theta Decoder
%   Theta-to-Reference Mapper
%   Force MPC / GRF optimizer
%   Mock WBC QP
%   Version-3 Impedance Residual
%
% Version-3 residual includes:
%   1) touchdown impact adaptation
%   2) stance force-mismatch adaptation
%   3) stance slip adaptation
%   4) mid-swing tracking stiffness adaptation
%
% IMPORTANT:
%   WBC and residual use mock Jacobians/foot states.
%   This is for interface + math structure verification.
%   Replace mock dynamics/Jacobians with URDF-derived quantities later.

clear; clc;

%% Example inputs
z_t = 0;  % 0 = conservative, 1 = aggressive

a_HL = [
    0.20;
   -0.30;
    0.40;
   -0.10
];

beta_t = [0.30; 0.40; 0.30];

x_hat = [
    0.00; 0.00; 0.42;
    0.03; -0.02; 0.10;
    0.25; 0.02; 0.00;
    0.00; 0.00; 0.05
];

u_cmd = [0.30; 0.00; 0.10];

%% Common parameters
params.dt = 0.02;
params.N  = 20;

params.robot.mass = 25.0;
params.robot.g    = 9.81;
params.robot.Ibody = diag([0.45, 1.20, 1.30]);

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

mpcParams.q0.x  = 0.0;
mpcParams.q0.y  = 0.0;
mpcParams.q0.z  = 80.0;
mpcParams.q0.roll  = 80.0;
mpcParams.q0.pitch = 80.0;
mpcParams.q0.yaw   = 1.0;
mpcParams.q0.vx = 40.0;
mpcParams.q0.vy = 40.0;
mpcParams.q0.vz = 30.0;
mpcParams.q0.wx = 1.0;
mpcParams.q0.wy = 1.0;
mpcParams.q0.wz = 30.0;

mpcParams.weightRange.wh = [0.5, 3.0];
mpcParams.weightRange.wv = [0.5, 3.0];
mpcParams.weightRange.we = [0.1, 5.0];
mpcParams.H_reg = 1e-8;

%% WBC parameters
wbcParams = struct();
wbcParams.dt = params.dt;
wbcParams.robot = params.robot;
wbcParams.fz_max_per_leg = mpcParams.fz_max_per_leg;
wbcParams.mu = [];
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

%% Impedance residual parameters, Version 3
impParams = struct();
impParams.lambda_res = 0.20;

impParams.phase_width = 0.10;
impParams.touchdown_phase = 1.0;

% Impact adaptation
impParams.alpha_imp = 25.0;  % damping up
impParams.beta_imp  = 20.0;  % stiffness down
impParams.s_imp0    = 0.25;

% Force-mismatch adaptation
impParams.alpha_F = 4.0;     % damping up
impParams.beta_F  = 3.0;     % stiffness down
impParams.F0      = 60.0;    % normalization [N]

% Normalize force mismatch by body weight per leg.
impParams.F0 = params.robot.mass * params.robot.g / 4;
impParams.F0 = max(impParams.F0, 30.0);

% Slip adaptation
impParams.alpha_slip = 2.0; %10.0; % damping up
impParams.beta_slip  = 2.0; %12.0; % tangential stiffness down
impParams.slip0      = 0.10; % [m/s]

% Mid-swing tracking adaptation
impParams.alpha_track = 15.0;
impParams.e0          = 0.03; % [m]

% Nominal impedance lookup tables.
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

% Mock current foot state.
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

IMP = impedanceResidualV3(z_t, Theta, Ref, WBC, MPC, impParams);

%% Display
fprintf("===== Version-3 impedance diagnostics =====\n");
disp("Ref.phase(:,1) = "); disp(Ref.phase(:,1));
disp("Ref.S(:,1) = "); disp(Ref.S(:,1));
disp("impact = "); disp(IMP.impact_signal);
disp("gate = "); disp(IMP.phase_gate);
disp("slip = "); disp(IMP.slip_signal);
disp("force mismatch = "); disp(IMP.force_mismatch);
disp("track error = "); disp(IMP.track_error);
disp("gamma = "); disp(IMP.gamma_leg);

%% Visualization
figure;
imagesc(Ref.S);
colormap(gray);
xlabel("Horizon step k");
ylabel("Leg index: 1 LF, 2 RF, 3 LH, 4 RH");
title("Contact schedule S_t, white=stance, black=swing");
colorbar;

figure;
plot(0:params.N-1, MPC.F_by_leg_z');
xlabel("Horizon step k");
ylabel("Vertical GRF f_z [N]");
legend("LF","RF","LH","RH");
title("Optimized vertical GRF over horizon");
grid on;

figure;
plot(0:params.N-1, sum(MPC.F_by_leg_z,1), "LineWidth", 1.5); hold on;
yline(params.robot.mass * params.robot.g, "--");
xlabel("Horizon step k");
ylabel("Total vertical GRF [N]");
legend("sum f_z", "mg");
title("Total vertical GRF vs body weight");
grid on;

figure;
bar([WBC.tau_OC, IMP.tau_res, IMP.tau_final]);
xlabel("Joint index");
ylabel("Torque [Nm]");
legend("tau OC", "tau residual", "tau final");
title("WBC torque + Version-3 impedance residual");
grid on;

figure;
bar([IMP.impact_signal, IMP.phase_gate, IMP.slip_signal, IMP.force_mismatch, IMP.track_error]);
xlabel("Leg index");
ylabel("Signal value");
% legend("impact", "gate", "slip", "force mismatch", "track error");
legend("impact", "gate", "slip", "force mismatch norm", "track error");
title("Version-3 residual signals");
grid on;

figure;
bar([IMP.Kp_delta_diag, IMP.Kd_delta_diag]);
xlabel("Leg index");
ylabel("Mean diagonal gain delta");
legend("mean diag Delta Kp", "mean diag Delta Kd");
title("Version-3 gain adaptation summary");
grid on;
