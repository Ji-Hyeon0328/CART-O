%% main_contact_mismatch_magnitude_sweep_demo.m
clear; clc; close all;

robotName = "go1";
Tsim      = 30;
levels = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00];

[params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd] = initBaseParams();
[params0, mpcParams0, wbcParams0, impParams0, x0, meta] = ...
    applyRobotPresetV3(robotName, params0, mpcParams0, wbcParams0, impParams0, x0);

cmd.z_t    = 0;
cmd.a_HL   = [0.35; 0.10; 0.45; 0.25];
cmd.beta_t = [0.40; 0.25; 0.35];

resNom = runSweepCase("nominal", 1.0, cmd, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, Tsim, meta);

resSlip = cell(1, numel(levels));
resSponge = cell(1, numel(levels));
for i = 1:numel(levels)
    resSlip{i}   = runSweepCase("slippery", levels(i), cmd, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, Tsim, meta);
    resSponge{i} = runSweepCase("sponge",   levels(i), cmd, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, Tsim, meta);
end

printSweepSummary(resNom, resSlip, resSponge, levels);
plotSweepMetrics(resNom, resSlip, resSponge, levels);
plotSweepSignals(resNom, resSlip, resSponge, levels);
plotSweepBodyResponse(resNom, resSlip, resSponge, levels);
plotRepresentativeTimeSeries(resNom, resSlip, resSponge, levels);

disp("Done. Variables: resNom, resSlip, resSponge, levels");

function [params, mpcParams, wbcParams, impParams, x_hat, u_cmd] = initBaseParams()
    x_hat = [0;0;0.42; 0.03;-0.02;0.10; 0.25;0.02;0; 0;0;0.05];
    u_cmd = [0.30; 0.00; 0.10];

    params.dt = 0.02;
    params.N  = 20;
    params.robot.mass  = 25;
    params.robot.g     = 9.81;
    params.robot.Ibody = diag([0.45, 1.20, 1.30]);

    params.hip_offset_body = [0.28,0.28,-0.28,-0.28; 0.16,-0.16,0.16,-0.16; 0,0,0,0];
    params.p_foot_now = [0.30,0.30,-0.28,-0.28; 0.18,-0.18,0.18,-0.18; 0,0,0,0];
    params.Kv_foot = diag([-0.08,-0.08,0]);

    mpcParams = struct();
    mpcParams.dt = params.dt;
    mpcParams.N  = params.N;
    mpcParams.robot = params.robot;
    mpcParams.fz_max_per_leg = 1.2 * params.robot.mass * params.robot.g;
    mpcParams.use_force_rate_bound = false;
    mpcParams.df_max = 500;
    mpcParams.rho_f  = 0.60;
    mpcParams.rho_df = 0.40;
    mpcParams.q0.x = 0; mpcParams.q0.y = 0; mpcParams.q0.z = 80;
    mpcParams.q0.roll = 80; mpcParams.q0.pitch = 80; mpcParams.q0.yaw = 1;
    mpcParams.q0.vx = 40; mpcParams.q0.vy = 40; mpcParams.q0.vz = 30;
    mpcParams.q0.wx = 1; mpcParams.q0.wy = 1; mpcParams.q0.wz = 30;
    mpcParams.weightRange.wh = [0.5,3.0];
    mpcParams.weightRange.wv = [0.5,3.0];
    mpcParams.weightRange.we = [0.1,5.0];
    mpcParams.H_reg = 1e-8;

    wbcParams = struct();
    wbcParams.dt = params.dt;
    wbcParams.robot = params.robot;
    wbcParams.fz_max_per_leg = mpcParams.fz_max_per_leg;
    wbcParams.mu = [];
    wbcParams.tau_min = -90 * ones(12,1);
    wbcParams.tau_max =  90 * ones(12,1);
    wbcParams.tau_prev = zeros(12,1);
    wbcParams.Kp_base = diag([40,40,80,80,80,30]);
    wbcParams.Kd_base = diag([10,10,20,20,20,8]);
    wbcParams.Kp_foot = diag([80,80,120]);
    wbcParams.Kd_foot = diag([10,10,15]);
    wbcParams.Wb = diag([20,20,100,100,100,20]);
    wbcParams.Wfoot = diag([50,50,100]);
    wbcParams.Wforce_per_foot = diag([1,1,3]);
    wbcParams.Wtau = 1e-3 * eye(12);
    wbcParams.Wdtau = 1e-2 * eye(12);
    wbcParams.H_reg = 1e-8;

    impParams = struct();
    impParams.lambda_res = 0.20;
    impParams.phase_width = 0.10;
    impParams.touchdown_phase = 1.0;
    impParams.alpha_imp = 25;
    impParams.beta_imp  = 20;
    impParams.s_imp0    = 0.25;
    impParams.alpha_F = 4;
    impParams.beta_F  = 3;
    impParams.F0      = 60;
    impParams.alpha_slip = 10;
    impParams.beta_slip  = 12;
    impParams.slip0      = 0.10;
    impParams.alpha_track = 15;
    impParams.e0          = 0.03;
    impParams.Kp.cons.st = diag([25,25,45]);
    impParams.Kd.cons.st = diag([18,18,30]);
    impParams.Kp.cons.sw = diag([60,60,90]);
    impParams.Kd.cons.sw = diag([7,7,10]);
    impParams.Kp.agg.st = diag([40,40,70]);
    impParams.Kd.agg.st = diag([20,20,35]);
    impParams.Kp.agg.sw = diag([90,90,130]);
    impParams.Kd.agg.sw = diag([10,10,15]);
    impParams.Kp_min = diag([5,5,5]);
    impParams.Kp_max = diag([200,200,250]);
    impParams.Kd_min = diag([1,1,1]);
    impParams.Kd_max = diag([80,80,100]);
    impParams.mock_position_error = zeros(3,4);
    impParams.mock_foot_velocity_now = zeros(3,4);
end

function out = runSweepCase(caseName, level, cmd, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, Tsim, meta)
    x_hat = x0;
    Theta0 = thetaDecoder(cmd.z_t, cmd.a_HL, x_hat, u_cmd);
    phaseState = Theta0.gait.phase_i;

    out.caseName = string(caseName);
    out.level = level;
    out.tau_final  = zeros(12, Tsim);
    out.fz_total   = zeros(1, Tsim);
    out.fz_target  = zeros(1, Tsim);
    out.impact     = zeros(4, Tsim);
    out.gate       = zeros(4, Tsim);
    out.slip       = zeros(4, Tsim);
    out.f_mis      = zeros(4, Tsim);
    out.track      = zeros(4, Tsim);
    out.body_h_err = zeros(1, Tsim);
    out.rollpitch_err = zeros(2, Tsim);

    for t = 1:Tsim
        params    = params0;
        mpcParams = mpcParams0;
        wbcParams = wbcParams0;
        impParams = impParams0;

        Theta = thetaDecoder(cmd.z_t, cmd.a_HL, x_hat, u_cmd);
        phaseState = mod(phaseState + params.dt / max(Theta.gait.T, 1e-6), 1.0);
        Theta.gait.phase_i = phaseState;
        Ref = thetaRefMapper(Theta, x_hat, u_cmd, params);

        MPC = forceMPC(x_hat, Ref, Theta, cmd.beta_t, mpcParams);
        WBC = wbcQP_robotScaledMock(x_hat, zeros(12,1), zeros(12,1), Ref, Theta, MPC, wbcParams);

        [mockPosErr, mockFootVel, bodyDrift] = mismatchInjection(caseName, level, t, Tsim);
        impParams.mock_position_error    = mockPosErr;
        impParams.mock_foot_velocity_now = mockFootVel;

        x_hat(3) = x_hat(3) + bodyDrift(1);
        x_hat(4) = x_hat(4) + bodyDrift(2);
        x_hat(5) = x_hat(5) + bodyDrift(3);

        IMP = impedanceResidualV3(cmd.z_t, Theta, Ref, WBC, MPC, impParams);

        out.tau_final(:,t) = IMP.tau_final;
        out.fz_total(t)    = sum(MPC.F_by_leg_z(:,1));
        out.fz_target(t)   = meta.mg;
        out.impact(:,t)    = IMP.impact_signal;
        out.gate(:,t)      = IMP.phase_gate;
        if isfield(IMP, 'slip_signal')
            out.slip(:,t) = IMP.slip_signal;
        elseif isfield(IMP, 'slip')
            out.slip(:,t) = IMP.slip;
        end
        if isfield(IMP, 'force_mismatch')
            out.f_mis(:,t) = IMP.force_mismatch;
        end
        if isfield(IMP, 'track_error')
            out.track(:,t) = IMP.track_error;
        end
        out.body_h_err(t) = Theta.base.h_body_ref - x_hat(3);
        out.rollpitch_err(:,t) = [Theta.base.roll_ref - x_hat(4); Theta.base.pitch_ref - x_hat(5)];

        x_hat(1) = x_hat(1) + params.dt * u_cmd(1);
        x_hat(2) = x_hat(2) + params.dt * u_cmd(2);
        x_hat(6) = x_hat(6) + params.dt * u_cmd(3);
        x_hat(3) = max(0.20, min(0.60, x_hat(3)));
        x_hat(4) = max(-0.25, min(0.25, x_hat(4)));
        x_hat(5) = max(-0.25, min(0.25, x_hat(5)));
    end

    tauNorm  = vecnorm(out.tau_final, 2, 1);
    dtauNorm = vecnorm(diff(out.tau_final, 1, 2), 2, 1);

    out.metric.torqueEffort  = sum(tauNorm.^2);
    out.metric.torqueRate    = sum(dtauNorm.^2);
    out.metric.touchImpact   = sum((out.gate .* out.impact).^2, 'all');
    out.metric.slipTotal     = sum(out.slip.^2, 'all');
    out.metric.forceMismatch = sum(out.f_mis.^2, 'all');
    out.metric.trackError    = sum(out.track.^2, 'all');
    out.metric.supportErr    = sum((out.fz_total - out.fz_target).^2);
    out.metric.meanBodyHErr  = mean(abs(out.body_h_err));
    out.metric.meanRPError   = mean(vecnorm(out.rollpitch_err, 2, 1));
end

function [mockPosErr, mockFootVel, bodyDrift] = mismatchInjection(caseName, level, t, Tsim)
    ph = 2*pi*(t-1)/max(Tsim,1);
    mockPosErr = level * [0.002*sin(ph), 0.002*cos(ph), 0.002*cos(ph), 0.002*sin(ph); ...
                         -0.001*cos(ph), 0.001*sin(ph), 0.001*cos(ph),-0.001*sin(ph); ...
                          0.000,         0.000,         0.000,         0.000];
    mockFootVel = zeros(3,4);
    bodyDrift = [0;0;0];

    switch lower(string(caseName))
        case "nominal"
            return
        case "slippery"
            mockPosErr(1,:) = level * [ 0.006, -0.005, 0.004, -0.004];
            mockPosErr(2,:) = level * [-0.003, 0.004, -0.004, 0.003];
            mockFootVel(1,:) = level * [ 0.050, -0.040, 0.060, -0.050];
            mockFootVel(2,:) = level * [-0.030,  0.035,-0.040,  0.030];
            bodyDrift = [0; level*0.004*cos(ph); -level*0.004*sin(ph)];
        case "sponge"
            mockPosErr(3,:) = level * ([-0.010, -0.007, -0.012, -0.008] + 0.002*[sin(ph), cos(ph), sin(2*ph), cos(2*ph)]);
            mockFootVel(3,:) = level * [-0.020, -0.015, -0.025, -0.018];
            bodyDrift = [level*(-0.0025); level*0.003*sin(ph); level*0.002*cos(ph)];
        otherwise
            error("Unknown case: %s", caseName);
    end
end

function printSweepSummary(resNom, resSlip, resSponge, levels)
    fprintf('\\n================ Mismatch Magnitude Sweep Summary ================\\n');
    fprintf('Nominal: tauEff=%.3e, tauRate=%.3e, impact=%.3e, slip=%.3e, bodyH=%.3e, RP=%.3e\\n', ...
        resNom.metric.torqueEffort, resNom.metric.torqueRate, resNom.metric.touchImpact, ...
        resNom.metric.slipTotal, resNom.metric.meanBodyHErr, resNom.metric.meanRPError);

    fprintf('\\n-- Slippery sweep --\\n');
    fprintf('%8s | %10s %10s %10s %10s %10s %10s\\n', 'level', 'tauEff', 'tauRate', 'impact', 'slip', 'hErr', 'RPerr');
    for i = 1:numel(levels)
        c = resSlip{i}.metric;
        fprintf('%8.2f | %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\\n', levels(i), c.torqueEffort, c.torqueRate, c.touchImpact, c.slipTotal, c.meanBodyHErr, c.meanRPError);
    end

    fprintf('\\n-- Sponge sweep --\\n');
    fprintf('%8s | %10s %10s %10s %10s %10s %10s\\n', 'level', 'tauEff', 'tauRate', 'impact', 'slip', 'hErr', 'RPerr');
    for i = 1:numel(levels)
        c = resSponge{i}.metric;
        fprintf('%8.2f | %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\\n', levels(i), c.torqueEffort, c.torqueRate, c.touchImpact, c.slipTotal, c.meanBodyHErr, c.meanRPError);
    end
end

function plotSweepMetrics(resNom, resSlip, resSponge, levels)
    nom = resNom.metric;
    slip_tauEff = zeros(size(levels)); slip_tauRate = zeros(size(levels)); slip_impact = zeros(size(levels)); slip_slip = zeros(size(levels)); slip_hErr = zeros(size(levels)); slip_rpErr = zeros(size(levels));
    spg_tauEff  = zeros(size(levels)); spg_tauRate  = zeros(size(levels)); spg_impact  = zeros(size(levels)); spg_slip  = zeros(size(levels)); spg_hErr  = zeros(size(levels)); spg_rpErr  = zeros(size(levels));
    for i = 1:numel(levels)
        c = resSlip{i}.metric; slip_tauEff(i)=c.torqueEffort; slip_tauRate(i)=c.torqueRate; slip_impact(i)=c.touchImpact; slip_slip(i)=c.slipTotal; slip_hErr(i)=c.meanBodyHErr; slip_rpErr(i)=c.meanRPError;
        c = resSponge{i}.metric; spg_tauEff(i)=c.torqueEffort; spg_tauRate(i)=c.torqueRate; spg_impact(i)=c.touchImpact; spg_slip(i)=c.slipTotal; spg_hErr(i)=c.meanBodyHErr; spg_rpErr(i)=c.meanRPError;
    end

    figure('Name','Mismatch magnitude sweep metrics');
    tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

    nexttile; hold on; plot(levels, slip_tauEff./nom.torqueEffort, '-o', 'LineWidth',1.5); plot(levels, spg_tauEff./nom.torqueEffort, '-s', 'LineWidth',1.5); yline(1,'--'); xlabel('mismatch level'); ylabel('ratio to nominal'); title('Torque effort'); legend('slippery','sponge','nominal','Location','best'); grid on;
    nexttile; hold on; plot(levels, slip_tauRate./nom.torqueRate, '-o', 'LineWidth',1.5); plot(levels, spg_tauRate./nom.torqueRate, '-s', 'LineWidth',1.5); yline(1,'--'); xlabel('mismatch level'); ylabel('ratio to nominal'); title('Torque-rate'); grid on;
    nexttile; hold on; plot(levels, safeRatio(slip_impact, nom.touchImpact), '-o', 'LineWidth',1.5); plot(levels, safeRatio(spg_impact, nom.touchImpact), '-s', 'LineWidth',1.5); yline(1,'--'); xlabel('mismatch level'); ylabel('ratio to nominal'); title('Touchdown impact'); grid on;
    nexttile; hold on; plot(levels, safeRatio(slip_slip, max(nom.slipTotal,1e-12)), '-o', 'LineWidth',1.5); plot(levels, safeRatio(spg_slip, max(nom.slipTotal,1e-12)), '-s', 'LineWidth',1.5); yline(1,'--'); xlabel('mismatch level'); ylabel('ratio to nominal'); title('Slip metric'); grid on;
    nexttile; hold on; plot(levels, slip_hErr./max(nom.meanBodyHErr,1e-12), '-o', 'LineWidth',1.5); plot(levels, spg_hErr./max(nom.meanBodyHErr,1e-12), '-s', 'LineWidth',1.5); yline(1,'--'); xlabel('mismatch level'); ylabel('ratio to nominal'); title('Body-height error'); grid on;
    nexttile; hold on; plot(levels, slip_rpErr./max(nom.meanRPError,1e-12), '-o', 'LineWidth',1.5); plot(levels, spg_rpErr./max(nom.meanRPError,1e-12), '-s', 'LineWidth',1.5); yline(1,'--'); xlabel('mismatch level'); ylabel('ratio to nominal'); title('Roll/pitch error'); grid on;
end

function plotSweepSignals(resNom, resSlip, resSponge, levels)
    nomImpact = mean(resNom.impact,2); nomSlip = mean(resNom.slip,2); nomFMis = mean(resNom.f_mis,2); nomTrack = mean(resNom.track,2);
    figure('Name','Mismatch signal sweep');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    nexttile; hold on;
    for leg=1:4
        valsSlip = zeros(size(levels)); valsSpg = zeros(size(levels));
        for i=1:numel(levels), valsSlip(i)=mean(resSlip{i}.impact(leg,:)); valsSpg(i)=mean(resSponge{i}.impact(leg,:)); end
        plot(levels, valsSlip, '-o', 'LineWidth',1.2); plot(levels, valsSpg, '--s', 'LineWidth',1.2);
    end
    yline(mean(nomImpact),'k--','nominal avg'); xlabel('mismatch level'); ylabel('mean impact'); title('Impact signal sweep'); grid on;

    nexttile; hold on;
    for leg=1:4
        valsSlip = zeros(size(levels)); valsSpg = zeros(size(levels));
        for i=1:numel(levels), valsSlip(i)=mean(resSlip{i}.slip(leg,:)); valsSpg(i)=mean(resSponge{i}.slip(leg,:)); end
        plot(levels, valsSlip, '-o', 'LineWidth',1.2); plot(levels, valsSpg, '--s', 'LineWidth',1.2);
    end
    yline(mean(nomSlip),'k--','nominal avg'); xlabel('mismatch level'); ylabel('mean slip'); title('Slip signal sweep'); grid on;

    nexttile; hold on;
    for leg=1:4
        valsSlip = zeros(size(levels)); valsSpg = zeros(size(levels));
        for i=1:numel(levels), valsSlip(i)=mean(resSlip{i}.f_mis(leg,:)); valsSpg(i)=mean(resSponge{i}.f_mis(leg,:)); end
        plot(levels, valsSlip, '-o', 'LineWidth',1.2); plot(levels, valsSpg, '--s', 'LineWidth',1.2);
    end
    yline(mean(nomFMis),'k--','nominal avg'); xlabel('mismatch level'); ylabel('mean force mismatch'); title('Force-mismatch sweep'); grid on;

    nexttile; hold on;
    for leg=1:4
        valsSlip = zeros(size(levels)); valsSpg = zeros(size(levels));
        for i=1:numel(levels), valsSlip(i)=mean(resSlip{i}.track(leg,:)); valsSpg(i)=mean(resSponge{i}.track(leg,:)); end
        plot(levels, valsSlip, '-o', 'LineWidth',1.2); plot(levels, valsSpg, '--s', 'LineWidth',1.2);
    end
    yline(mean(nomTrack),'k--','nominal avg'); xlabel('mismatch level'); ylabel('mean track err'); title('Tracking-error sweep'); grid on;
end

function plotSweepBodyResponse(resNom, resSlip, resSponge, levels)
    figure('Name','Body/support response sweep');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    nexttile; hold on;
    for i=1:numel(levels), plot(resSlip{i}.fz_total, 'LineWidth', 1.0); end
    yline(resNom.fz_target(1),'k--','mg'); xlabel('step'); ylabel('\Sigma f_z [N]'); title('Slippery: support force'); grid on;

    nexttile; hold on;
    for i=1:numel(levels), plot(resSponge{i}.fz_total, 'LineWidth', 1.0); end
    yline(resNom.fz_target(1),'k--','mg'); xlabel('step'); ylabel('\Sigma f_z [N]'); title('Sponge: support force'); grid on;

    nexttile; hold on;
    for i=1:numel(levels), plot(resSlip{i}.body_h_err, 'LineWidth', 1.2); end
    xlabel('step'); ylabel('h body err [m]'); title('Slippery: body-height error'); grid on;

    nexttile; hold on;
    for i=1:numel(levels), plot(resSponge{i}.body_h_err, 'LineWidth', 1.2); end
    xlabel('step'); ylabel('h body err [m]'); title('Sponge: body-height error'); grid on;
end

function plotRepresentativeTimeSeries(resNom, resSlip, resSponge, levels)
    [~, idxMid] = min(abs(levels - 1.0)); idxMax = numel(levels);
    figure('Name','Representative time-series');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    nexttile; hold on; plot(vecnorm(resNom.tau_final,2,1),'k','LineWidth',1.5); plot(vecnorm(resSlip{idxMid}.tau_final,2,1),'LineWidth',1.3); plot(vecnorm(resSlip{idxMax}.tau_final,2,1),'LineWidth',1.3); xlabel('step'); ylabel('||tau||'); title('Slippery torque norm'); legend('nominal','slip mid','slip max','Location','best'); grid on;
    nexttile; hold on; plot(vecnorm(resNom.tau_final,2,1),'k','LineWidth',1.5); plot(vecnorm(resSponge{idxMid}.tau_final,2,1),'LineWidth',1.3); plot(vecnorm(resSponge{idxMax}.tau_final,2,1),'LineWidth',1.3); xlabel('step'); ylabel('||tau||'); title('Sponge torque norm'); legend('nominal','spg mid','spg max','Location','best'); grid on;
    nexttile; hold on; plot(vecnorm(resNom.rollpitch_err,2,1),'k','LineWidth',1.5); plot(vecnorm(resSlip{idxMid}.rollpitch_err,2,1),'LineWidth',1.3); plot(vecnorm(resSlip{idxMax}.rollpitch_err,2,1),'LineWidth',1.3); xlabel('step'); ylabel('RP err'); title('Slippery RP error'); grid on;
    nexttile; hold on; plot(vecnorm(resNom.rollpitch_err,2,1),'k','LineWidth',1.5); plot(vecnorm(resSponge{idxMid}.rollpitch_err,2,1),'LineWidth',1.3); plot(vecnorm(resSponge{idxMax}.rollpitch_err,2,1),'LineWidth',1.3); xlabel('step'); ylabel('RP err'); title('Sponge RP error'); grid on;
end

function r = safeRatio(a, b)
    r = a ./ max(b, 1e-12);
end
