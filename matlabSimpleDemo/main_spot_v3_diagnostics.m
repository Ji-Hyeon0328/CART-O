%% main_spot_v3_diagnostics.m
% Spot-only diagnostic run for Version-3 impedance residual.
%
% Required files in MATLAB path:
%   thetaDecoder.m
%   thetaRefMapper.m
%   forceMPC.m
%   impedanceResidualV3.m
%   applyRobotPresetV3.m
%   wbcQP_robotScaledMock.m
%
% This is a scale-aware mock demo, not true Spot URDF WBC.

clear; clc; close all;

robotName = "spot";

%% High-level inputs
z_t = 0;  % conservative

a_HL = [
    0.20;
   -0.20;
    0.30;
   -0.10
];

beta_t = [0.30; 0.40; 0.30];

x_hat = [
    0.00; 0.00; 0.55;
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

%% Version-3 impedance residual parameters
impParams = struct();
impParams.lambda_res = 0.20;

impParams.phase_width = 0.10;
impParams.touchdown_phase = 1.0;

impParams.alpha_imp = 25.0;
impParams.beta_imp  = 20.0;
impParams.s_imp0    = 0.25;

impParams.alpha_F = 4.0;
impParams.beta_F  = 3.0;
impParams.F0      = 60.0;

impParams.alpha_slip = 10.0;
impParams.beta_slip  = 12.0;
impParams.slip0      = 0.10;

impParams.alpha_track = 15.0;
impParams.e0          = 0.03;

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

%% Apply Spot preset
[params, mpcParams, wbcParams, impParams, x_hat, meta] = ...
    applyRobotPresetV3(robotName, params, mpcParams, wbcParams, impParams, x_hat);

%% Run pipeline
Theta = thetaDecoder(z_t, a_HL, x_hat, u_cmd);
Ref   = thetaRefMapper(Theta, x_hat, u_cmd, params);
MPC   = forceMPC(x_hat, Ref, Theta, beta_t, mpcParams);

qj  = zeros(12,1);
dqj = zeros(12,1);
WBC = wbcQP_robotScaledMock(x_hat, qj, dqj, Ref, Theta, MPC, wbcParams);

IMP = impedanceResidualV3(z_t, Theta, Ref, WBC, MPC, impParams);

%% Diagnostics
Fz_total = sum(MPC.F_by_leg_z,1);
tau_ratio = abs(IMP.tau_final) ./ max(wbcParams.tau_max, 1e-9);

fprintf("\n===== Spot Version-3 Diagnostics =====\n");
fprintf("mass = %.3f kg\n", meta.mass);
fprintf("mg = %.3f N\n", meta.mg);
fprintf("F0 = %.3f N\n", impParams.F0);
fprintf("nominal body height = %.3f m\n", meta.nominal_body_height);
fprintf("MPC exitflag = %d\n", MPC.exitflag);
fprintf("WBC exitflag = %d\n", WBC.exitflag);
fprintf("mean(sum fz) = %.3f N\n", mean(Fz_total));
fprintf("mean(sum fz / mg) = %.3f\n", mean(Fz_total / meta.mg));
fprintf("max(abs(tau_final ./ tau_max)) = %.3f\n", max(tau_ratio));
fprintf("mean(abs(tau_final)) = %.3f Nm\n", mean(abs(IMP.tau_final)));

disp("Ref.phase(:,1) = "); disp(Ref.phase(:,1));
disp("Ref.S(:,1) = "); disp(Ref.S(:,1));
disp("Normalized force mismatch = "); disp(IMP.force_mismatch);
if isfield(IMP, "force_mismatch_raw")
    disp("Raw force mismatch [N] = "); disp(IMP.force_mismatch_raw);
end
disp("Kp delta diag = "); disp(IMP.Kp_delta_diag);
disp("Kd delta diag = "); disp(IMP.Kd_delta_diag);
disp("tau_final = "); disp(IMP.tau_final);
disp("tau_final / tau_max = "); disp(tau_ratio);

%% Plots
figure;
imagesc(Ref.S);
colormap(gray);
xlabel("Horizon step k");
ylabel("Leg index: 1 LF, 2 RF, 3 LH, 4 RH");
title("Spot contact schedule S_t, white=stance, black=swing");
colorbar;

figure;
plot(0:params.N-1, MPC.F_by_leg_z');
xlabel("Horizon step k");
ylabel("Vertical GRF f_z [N]");
legend("LF","RF","LH","RH");
title("Spot optimized vertical GRF over horizon");
grid on;

figure;
plot(0:params.N-1, Fz_total, "LineWidth", 1.5); hold on;
yline(meta.mg, "--", "mg");
xlabel("Horizon step k");
ylabel("Total vertical GRF [N]");
legend("sum f_z", "mg");
title("Spot total vertical GRF vs body weight");
grid on;

figure;
bar([WBC.tau_OC, IMP.tau_res, IMP.tau_final]);
xlabel("Joint index");
ylabel("Torque [Nm]");
legend("tau OC", "tau residual", "tau final");
title("Spot WBC torque + Version-3 impedance residual");
grid on;

figure;
bar(tau_ratio);
xlabel("Joint index");
ylabel("|tau final| / tau max");
title("Spot torque limit usage ratio");
grid on;
yline(1.0, "--", "limit");

figure;
bar([IMP.impact_signal, IMP.phase_gate, IMP.slip_signal, IMP.force_mismatch, IMP.track_error]);
xlabel("Leg index");
ylabel("Signal value");
legend("impact", "gate", "slip", "force mismatch norm", "track error");
title("Spot Version-3 residual signals");
grid on;

figure;
bar([IMP.Kp_delta_diag, IMP.Kd_delta_diag]);
xlabel("Leg index");
ylabel("Mean diagonal gain delta");
legend("mean diag Delta Kp", "mean diag Delta Kd");
title("Spot Version-3 gain adaptation summary");
grid on;
