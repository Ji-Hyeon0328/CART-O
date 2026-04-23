%% main_virtual_terrain_lowlevel_demo.m
% Virtual terrain demo for CARTO low-level controller.
%
% This script emulates high-level planner outputs for:
%   flat / rough / up_slope / down_slope
%
% Baseline:
%   Fixed flat high-level command for all terrain cases.
%   No impedance residual:
%       tau_base = tau_OC
%
% Proposed:
%   Terrain-conditioned z_t, a_HL, beta_t.
%   Version-3 impedance residual:
%       tau_ours = tau_OC + lambda_res * tau_res
%
% Energy-efficiency note:
%   This is not true CoT yet. It is a low-level surrogate demo using:
%       1) sum ||tau||^2
%       2) sum ||Delta tau||^2
%       3) sum ||f_z||^2 and sum ||Delta f_z||^2 over MPC horizon
%       4) touchdown impact surrogate sum gate*impact^2
%
% Required files in MATLAB path:
%   thetaDecoder.m
%   thetaRefMapper.m
%   forceMPC.m
%   impedanceResidualV3.m
%   applyRobotPresetV3.m
%   wbcQP_robotScaledMock.m

clear; clc; close all;

robotName = "go1";      % recommended first: "go1"; also "spot" or "toy25"
Tsim = 40;
terrainList = ["flat", "rough", "up_slope", "down_slope"];

[params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd] = initBaseParams();

[params0, mpcParams0, wbcParams0, impParams0, x0, meta] = ...
    applyRobotPresetV3(robotName, params0, mpcParams0, wbcParams0, impParams0, x0);

fprintf("\n===== Virtual Terrain Low-Level Demo =====\n");
fprintf("Robot preset: %s | mass %.2f kg | mg %.2f N\n", robotName, meta.mass, meta.mg);

results = struct();

for terrain = terrainList
    key = matlab.lang.makeValidName(terrain);

    baselineCmd = virtualTerrainHighLevelCommand("baseline_flat");
    proposedCmd = virtualTerrainHighLevelCommand(terrain);

    baselineOut = runVirtualSequence(terrain, baselineCmd, false, ...
        Tsim, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, meta);

    proposedOut = runVirtualSequence(terrain, proposedCmd, true, ...
        Tsim, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, meta);

    results.(key).baseline = baselineOut;
    results.(key).proposed = proposedOut;
    results.(key).cmd_baseline = baselineCmd;
    results.(key).cmd_proposed = proposedCmd;

    printMetricSummary(terrain, baselineOut.metrics, proposedOut.metrics);
end

plotThetaSummary(results, terrainList);
plotEnergyMetricSummary(results, terrainList);
plotTerrainDetailed(results, terrainList, meta);

disp("Demo complete. Results are stored in variable: results");

%% ============================================================
% Local functions
% ============================================================

function [params, mpcParams, wbcParams, impParams, x_hat, u_cmd] = initBaseParams()
    x_hat = [
        0.00; 0.00; 0.42;
        0.03; -0.02; 0.10;
        0.25; 0.02; 0.00;
        0.00; 0.00; 0.05
    ];

    u_cmd = [0.30; 0.00; 0.10];

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

    impParams.mock_position_error = zeros(3,4);
    impParams.mock_foot_velocity_now = zeros(3,4);
end

function cmd = virtualTerrainHighLevelCommand(terrain)
    terrain = lower(string(terrain));

    switch terrain
        case "baseline_flat"
            cmd.z_t = 0;
            cmd.a_HL = [0.0; 0.0; 0.0; 0.0];
            cmd.beta_t = [0.33; 0.34; 0.33];
            cmd.label = "baseline fixed flat";

        case "flat"
            cmd.z_t = 1;
            cmd.a_HL = [0.0; 0.0; -0.1; -0.1];
            cmd.beta_t = [0.25; 0.45; 0.30];
            cmd.label = "flat velocity-friendly";

        case "rough"
            cmd.z_t = 0;
            cmd.a_HL = [0.7; 0.1; 0.6; 0.3];
            cmd.beta_t = [0.45; 0.20; 0.35];
            cmd.label = "rough clearance/stability";

        case "up_slope"
            cmd.z_t = 0;
            cmd.a_HL = [0.4; -0.1; 0.5; 0.2];
            cmd.beta_t = [0.40; 0.25; 0.35];
            cmd.label = "up slope stable support";

        case "down_slope"
            cmd.z_t = 0;
            cmd.a_HL = [0.2; -0.2; 0.6; 0.5];
            cmd.beta_t = [0.35; 0.20; 0.45];
            cmd.label = "down slope damping/energy";

        otherwise
            error("Unknown terrain: %s", terrain);
    end
end

function out = runVirtualSequence(terrain, cmd, useResidual, Tsim, ...
    params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, meta)

    tau_OC_hist = zeros(12,Tsim);
    tau_final_hist = zeros(12,Tsim);
    tau_res_hist = zeros(12,Tsim);
    fz_total_hist = zeros(1,Tsim);
    force_effort_hist = zeros(1,Tsim);
    force_rate_effort_hist = zeros(1,Tsim);
    impact_hist = zeros(4,Tsim);
    gate_hist = zeros(4,Tsim);
    force_mismatch_hist = zeros(4,Tsim);
    kdes_hist = zeros(1,Tsim);
    duty_hist = zeros(4,Tsim);
    hswing_hist = zeros(4,Tsim);
    body_h_hist = zeros(1,Tsim);
    mpc_weights = zeros(3,Tsim);

    contact_first = [];

    x_hat = x0;
    Theta0 = thetaDecoder(cmd.z_t, cmd.a_HL, x_hat, u_cmd);
    phase0 = Theta0.gait.phase_i;

    for t = 1:Tsim
        params = params0;
        mpcParams = mpcParams0;
        wbcParams = wbcParams0;
        impParams = impParams0;

        [mockPosErr, mockFootVel] = virtualTerrainDisturbance(terrain, t, Tsim);
        impParams.mock_position_error = mockPosErr;
        impParams.mock_foot_velocity_now = mockFootVel;

        Theta = thetaDecoder(cmd.z_t, cmd.a_HL, x_hat, u_cmd);
        phaseAdvance = (t-1) * params.dt / max(Theta.gait.T, 1e-6);
        Theta.gait.phase_i = mod(phase0 + phaseAdvance, 1.0);

        Ref = thetaRefMapper(Theta, x_hat, u_cmd, params);
        MPC = forceMPC(x_hat, Ref, Theta, cmd.beta_t, mpcParams);

        qj  = zeros(12,1);
        dqj = zeros(12,1);
        WBC = wbcQP_robotScaledMock(x_hat, qj, dqj, Ref, Theta, MPC, wbcParams);

        IMP = impedanceResidualV3(cmd.z_t, Theta, Ref, WBC, MPC, impParams);

        tau_OC = WBC.tau_OC;
        tau_res = IMP.tau_res;

        if useResidual
            tau_final = IMP.tau_final;
        else
            tau_res = zeros(size(tau_res));
            tau_final = tau_OC;
        end

        tau_OC_hist(:,t) = tau_OC;
        tau_res_hist(:,t) = tau_res;
        tau_final_hist(:,t) = tau_final;

        fz_total_hist(t) = sum(MPC.F_by_leg_z(:,1));

        [Ef, Edf] = forceHorizonSurrogates(MPC, meta);
        force_effort_hist(t) = Ef;
        force_rate_effort_hist(t) = Edf;

        impact_hist(:,t) = IMP.impact_signal;
        gate_hist(:,t) = IMP.phase_gate;
        force_mismatch_hist(:,t) = IMP.force_mismatch;

        kdes_hist(t) = Theta.ctrl.k_des;
        duty_hist(:,t) = Theta.gait.duty_i;
        hswing_hist(:,t) = Theta.foot.h_swing_i;
        body_h_hist(t) = Theta.base.h_body_ref;

        if isfield(MPC, "weights")
            mpc_weights(:,t) = [MPC.weights.wh; MPC.weights.wv; MPC.weights.we];
        end

        if isempty(contact_first)
            contact_first = Ref.S;
        end

        x_hat(1) = x_hat(1) + params.dt * u_cmd(1);
        x_hat(2) = x_hat(2) + params.dt * u_cmd(2);
        x_hat(6) = x_hat(6) + params.dt * u_cmd(3);
    end

    out.tau_OC = tau_OC_hist;
    out.tau_res = tau_res_hist;
    out.tau_final = tau_final_hist;
    out.fz_total = fz_total_hist;
    out.force_effort = force_effort_hist;
    out.force_rate_effort = force_rate_effort_hist;
    out.impact = impact_hist;
    out.gate = gate_hist;
    out.force_mismatch = force_mismatch_hist;
    out.kdes = kdes_hist;
    out.duty = duty_hist;
    out.hswing = hswing_hist;
    out.body_h = body_h_hist;
    out.mpc_weights = mpc_weights;
    out.contact_first = contact_first;
    out.cmd = cmd;
    out.metrics = computeMetrics(out, meta);
end

function [posErr, footVel] = virtualTerrainDisturbance(terrain, t, Tsim)
    terrain = lower(string(terrain));
    phase = 2*pi*(t-1)/max(Tsim,1);

    posErr = [
        0.010, -0.008,  0.006,  0.004;
       -0.004,  0.004,  0.006, -0.006;
        0.000,  0.000, -0.004, -0.003
    ];

    footVel = [
        0.010,  0.000,  0.010, -0.010;
        0.000,  0.008, -0.008,  0.000;
       -0.030, -0.020, -0.080, -0.050
    ];

    switch terrain
        case "flat"
            scaleImp = 0.7; scaleSlip = 0.8;

        case "rough"
            scaleImp = 1.8; scaleSlip = 1.6;
            posErr = posErr + [
                0.004*sin(phase), 0, 0.004*cos(phase), 0;
                0, 0.004*cos(phase), 0, 0.004*sin(phase);
                0.006, -0.004, 0.008, -0.005
            ];

        case "up_slope"
            scaleImp = 1.1; scaleSlip = 1.1;
            posErr(3,:) = posErr(3,:) + [0.003, 0.003, -0.002, -0.002];

        case "down_slope"
            scaleImp = 1.6; scaleSlip = 1.4;
            footVel(3,:) = footVel(3,:) + [-0.04, -0.02, -0.04, -0.03];

        otherwise
            scaleImp = 1.0; scaleSlip = 1.0;
    end

    footVel(1:2,:) = scaleSlip * footVel(1:2,:);
    footVel(3,:)   = scaleImp  * footVel(3,:);
end

function [Ef, Edf] = forceHorizonSurrogates(MPC, meta)
    Fz = MPC.F_by_leg_z;
    fscale = max(meta.mg, 1e-6);
    Ef = sum(Fz(:).^2) / (fscale^2);
    dF = diff(Fz,1,2);
    Edf = sum(dF(:).^2) / (fscale^2);
end

function metrics = computeMetrics(out, meta)
    tau = out.tau_final;
    tau_norm = vecnorm(tau,2,1);
    dtau = diff(tau,1,2);
    dtau_norm = vecnorm(dtau,2,1);

    metrics.torque_effort = sum(tau_norm.^2);
    metrics.torque_rate_effort = sum(dtau_norm.^2);
    metrics.force_effort = sum(out.force_effort);
    metrics.force_rate_effort = sum(out.force_rate_effort);

    tdImpact = out.gate .* out.impact;
    metrics.touchdown_impact = sum(tdImpact(:).^2);
    metrics.mean_force_mismatch = mean(out.force_mismatch(:));
    metrics.mean_fz_ratio = mean(out.fz_total / max(meta.mg,1e-6));
    metrics.mean_tau_norm = mean(tau_norm);
    metrics.max_tau_norm = max(tau_norm);
end

function printMetricSummary(terrain, base, ours)
    fprintf("\nTerrain: %s\n", terrain);
    fprintf("  mean fz/mg:         base %.3f | ours %.3f\n", base.mean_fz_ratio, ours.mean_fz_ratio);
    fprintf("  torque effort:      base %.3e | ours %.3e | ratio %.3f\n", ...
        base.torque_effort, ours.torque_effort, safeRatio(ours.torque_effort, base.torque_effort));
    fprintf("  torque-rate effort: base %.3e | ours %.3e | ratio %.3f\n", ...
        base.torque_rate_effort, ours.torque_rate_effort, safeRatio(ours.torque_rate_effort, base.torque_rate_effort));
    fprintf("  force effort:       base %.3e | ours %.3e | ratio %.3f\n", ...
        base.force_effort, ours.force_effort, safeRatio(ours.force_effort, base.force_effort));
    fprintf("  force-rate effort:  base %.3e | ours %.3e | ratio %.3f\n", ...
        base.force_rate_effort, ours.force_rate_effort, safeRatio(ours.force_rate_effort, base.force_rate_effort));
    fprintf("  touchdown impact:   base %.3e | ours %.3e | ratio %.3f\n", ...
        base.touchdown_impact, ours.touchdown_impact, safeRatio(ours.touchdown_impact, base.touchdown_impact));
end

function r = safeRatio(a,b)
    r = a / max(b, 1e-12);
end

function plotThetaSummary(results, terrainList)
    n = numel(terrainList);
    body_h = zeros(n,2); duty = zeros(n,2); hswing = zeros(n,2); kdes = zeros(n,2);
    wh = zeros(n,1); wv = zeros(n,1); we = zeros(n,1);

    for i = 1:n
        key = matlab.lang.makeValidName(terrainList(i));
        b = results.(key).baseline;
        o = results.(key).proposed;

        body_h(i,:) = [mean(b.body_h), mean(o.body_h)];
        duty(i,:)   = [mean(b.duty(:)), mean(o.duty(:))];
        hswing(i,:) = [mean(b.hswing(:)), mean(o.hswing(:))];
        kdes(i,:)   = [mean(b.kdes), mean(o.kdes)];

        wh(i) = mean(o.mpc_weights(1,:));
        wv(i) = mean(o.mpc_weights(2,:));
        we(i) = mean(o.mpc_weights(3,:));
    end

    figure;
    bar(categorical(terrainList), [body_h(:,1), body_h(:,2), hswing(:,1), hswing(:,2)]);
    ylabel("Reference value [m]");
    legend("base body h", "ours body h", "base swing h", "ours swing h");
    title("Decoded base height and swing clearance");
    grid on;

    figure;
    bar(categorical(terrainList), [duty(:,1), duty(:,2), kdes(:,1), kdes(:,2)]);
    ylabel("Decoded value");
    legend("base duty", "ours duty", "base k des", "ours k des");
    title("Decoded duty factor and impedance level");
    grid on;

    figure;
    bar(categorical(terrainList), [wh, wv, we]);
    ylabel("MPC weight scale");
    legend("w_h", "w_v", "w_e");
    title("Proposed beta-conditioned MPC weights");
    grid on;
end

function plotEnergyMetricSummary(results, terrainList)
    n = numel(terrainList);
    M = zeros(n,5);

    for i = 1:n
        key = matlab.lang.makeValidName(terrainList(i));
        b = results.(key).baseline.metrics;
        o = results.(key).proposed.metrics;

        M(i,1) = safeRatio(o.torque_effort, b.torque_effort);
        M(i,2) = safeRatio(o.torque_rate_effort, b.torque_rate_effort);
        M(i,3) = safeRatio(o.force_effort, b.force_effort);
        M(i,4) = safeRatio(o.force_rate_effort, b.force_rate_effort);
        M(i,5) = safeRatio(o.touchdown_impact, b.touchdown_impact);
    end

    figure;
    bar(categorical(terrainList), M);
    yline(1.0, "--", "baseline");
    ylabel("Ours / Baseline ratio");
    legend("torque effort", "torque-rate", "force effort", "force-rate", "touchdown impact");
    title("Energy/impact surrogate ratios, lower is better");
    grid on;
end

function plotTerrainDetailed(results, terrainList, meta)
    for i = 1:numel(terrainList)
        terrain = terrainList(i);
        key = matlab.lang.makeValidName(terrain);

        figure;
        imagesc(results.(key).proposed.contact_first);
        colormap(gray);
        xlabel("Horizon step k");
        ylabel("Leg index: 1 LF, 2 RF, 3 LH, 4 RH");
        title("Proposed contact schedule: " + terrain);
        colorbar;
    end

    figure;
    hold on;
    for i = 1:numel(terrainList)
        key = matlab.lang.makeValidName(terrainList(i));
        tauO = results.(key).proposed.tau_final;
        plot(vecnorm(tauO,2,1), "LineWidth", 1.5);
    end
    xlabel("Virtual control step");
    ylabel("||tau final||_2 [Nm]");
    legend(terrainList);
    title("Proposed torque norm over virtual terrain sequence");
    grid on;
end
