%% main_compare_robot_presets_v3.m
% Compare Version-3 controller demo between:
%   1) toy25 original demo robot
%   2) Unitree Go1 approximate preset
%
% Required files in same folder/path:
%   thetaDecoder.m
%   thetaRefMapper.m
%   forceMPC.m
%   wbcQP_mock.m
%   impedanceResidualV3.m
%   applyRobotPresetV3.m
%
% This script swaps robot physical parameters and dependent controller scales.

clear; clc; close all;

robotList = ["toy25", "go1"];
results = struct();

for ri = 1:numel(robotList)
    robotName = robotList(ri);
    fprintf("\n==============================\n");
    fprintf("Running robot preset: %s\n", robotName);
    fprintf("==============================\n");

    %% Base demo inputs
    z_t = 0;  % 0 conservative, 1 aggressive
    a_HL = [0.20; -0.30; 0.40; -0.10];
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

    impParams.alpha_imp = 25.0;
    impParams.beta_imp  = 20.0;
    impParams.s_imp0    = 0.25;

    impParams.alpha_F = 4.0;
    impParams.beta_F  = 3.0;
    impParams.F0      = 60.0;  % overwritten by robot preset

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

    %% Apply robot-specific preset and dependent scaling
    [params, mpcParams, wbcParams, impParams, x_hat, meta] = ...
        applyRobotPresetV3(robotName, params, mpcParams, wbcParams, impParams, x_hat);

    %% Run pipeline
    Theta = thetaDecoder(z_t, a_HL, x_hat, u_cmd);
    Ref   = thetaRefMapper(Theta, x_hat, u_cmd, params);
    MPC   = forceMPC(x_hat, Ref, Theta, beta_t, mpcParams);

    qj  = zeros(12,1);
    dqj = zeros(12,1);
    WBC = wbcQP_mock(x_hat, qj, dqj, Ref, Theta, MPC, wbcParams);

    IMP = impedanceResidualV3(z_t, Theta, Ref, WBC, MPC, impParams);

    %% Store
    key = matlab.lang.makeValidName(robotName);
    results.(key).meta = meta;
    results.(key).Theta = Theta;
    results.(key).Ref = Ref;
    results.(key).MPC = MPC;
    results.(key).WBC = WBC;
    results.(key).IMP = IMP;
    results.(key).params = params;
    results.(key).mpcParams = mpcParams;
    results.(key).wbcParams = wbcParams;
    results.(key).impParams = impParams;

    %% Print summary
    fprintf("mass = %.3f kg, mg = %.3f N, F0 = %.3f N\n", meta.mass, meta.mg, meta.F0);
    fprintf("MPC exitflag = %d, WBC exitflag = %d\n", MPC.exitflag, WBC.exitflag);
    fprintf("Total vertical GRF mean = %.3f N\n", mean(sum(MPC.F_by_leg_z,1)));
    fprintf("Mean tau_final abs = %.3f Nm\n", mean(abs(IMP.tau_final)));
    fprintf("Force mismatch norm = "); disp(IMP.force_mismatch');
    fprintf("Kp delta diag = "); disp(IMP.Kp_delta_diag');
    fprintf("Kd delta diag = "); disp(IMP.Kd_delta_diag');
end

%% Comparison plots
toy = results.toy25;
go1 = results.go1;

figure;
plot(0:toy.params.N-1, sum(toy.MPC.F_by_leg_z,1), "LineWidth", 1.5); hold on;
plot(0:go1.params.N-1, sum(go1.MPC.F_by_leg_z,1), "LineWidth", 1.5);
yline(toy.meta.mg, "--", "toy25 mg");
yline(go1.meta.mg, "--", "go1 mg");
xlabel("Horizon step k");
ylabel("Total vertical GRF [N]");
legend("toy25 sum f_z", "go1 sum f_z", "toy25 mg", "go1 mg");
title("Total vertical GRF comparison");
grid on;

figure;
bar([toy.IMP.Kp_delta_diag, toy.IMP.Kd_delta_diag, go1.IMP.Kp_delta_diag, go1.IMP.Kd_delta_diag]);
xlabel("Leg index");
ylabel("Mean diagonal gain delta");
legend("toy Kp", "toy Kd", "go1 Kp", "go1 Kd");
title("Version-3 gain adaptation: toy25 vs Go1");
grid on;

figure;
bar([toy.IMP.tau_final, go1.IMP.tau_final]);
xlabel("Joint index");
ylabel("Final torque [Nm]");
legend("toy25 tau final", "go1 tau final");
title("Final torque comparison");
grid on;

figure;
bar([toy.IMP.force_mismatch, go1.IMP.force_mismatch]);
xlabel("Leg index");
ylabel("Normalized force mismatch");
legend("toy25", "go1");
title("Normalized force mismatch comparison");
grid on;

disp("Comparison complete. Results are stored in variable: results");
