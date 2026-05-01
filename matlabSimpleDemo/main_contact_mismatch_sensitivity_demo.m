%% main_contact_mismatch_sensitivity_demo.m
clear; clc; close all;

robotName = "go1";      % "go1" or "spot"
Tsim      = 30;
showPrint = true;

cases = ["nominal", "sponge", "slippery", "mud_slope"];

[params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd] = initBaseParams();
[params0, mpcParams0, wbcParams0, impParams0, x0, meta] = ...
    applyRobotPresetV3(robotName, params0, mpcParams0, wbcParams0, impParams0, x0);

cmd.z_t    = 0;
cmd.a_HL   = [0.35; 0.10; 0.45; 0.25];
cmd.beta_t = [0.40; 0.25; 0.35];

results = struct();
for i = 1:numel(cases)
    caseName = cases(i);
    results.(caseName) = runMismatchCase(caseName, cmd, ...
        params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, Tsim, meta);
end

if showPrint
    printCaseSummary(results, cases);
end

plotCaseMetrics(results, cases);
plotCaseSignals(results, cases);
plotCaseTorques(results, cases);
plotCaseSupport(results, cases);
plotCaseGainAdaptation(results, cases);

disp("Done. Result struct: results");

function [params, mpcParams, wbcParams, impParams, x_hat, u_cmd] = initBaseParams()
    x_hat = [0;0;0.42; 0.03;-0.02;0.10; 0.25;0.02;0; 0;0;0.05];
    u_cmd = [0.30; 0.00; 0.10];

    params.dt = 0.02;
    params.N  = 20;

    params.robot.mass  = 25;
    params.robot.g     = 9.81;
    params.robot.Ibody = diag([0.45, 1.20, 1.30]);

    params.hip_offset_body = [ ...
         0.28,  0.28, -0.28, -0.28; ...
         0.16, -0.16,  0.16, -0.16; ...
         0.00,  0.00,  0.00,  0.00];
    params.p_foot_now = [ ...
         0.30,  0.30, -0.28, -0.28; ...
         0.18, -0.18,  0.18, -0.18; ...
         0.00,  0.00,  0.00,  0.00];
    params.Kv_foot = diag([-0.08, -0.08, 0]);

    mpcParams = struct();
    mpcParams.dt = params.dt;
    mpcParams.N  = params.N;
    mpcParams.robot = params.robot;
    mpcParams.fz_max_per_leg = 1.2 * params.robot.mass * params.robot.g;
    mpcParams.use_force_rate_bound = false;
    mpcParams.df_max = 500;
    mpcParams.rho_f  = 0.60;
    mpcParams.rho_df = 0.40;
    mpcParams.q0.x     = 0;
    mpcParams.q0.y     = 0;
    mpcParams.q0.z     = 80;
    mpcParams.q0.roll  = 80;
    mpcParams.q0.pitch = 80;
    mpcParams.q0.yaw   = 1;
    mpcParams.q0.vx    = 40;
    mpcParams.q0.vy    = 40;
    mpcParams.q0.vz    = 30;
    mpcParams.q0.wx    = 1;
    mpcParams.q0.wy    = 1;
    mpcParams.q0.wz    = 30;
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

function out = runMismatchCase(caseName, cmd, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, Tsim, meta)
    x_hat = x0;
    Theta0 = thetaDecoder(cmd.z_t, cmd.a_HL, x_hat, u_cmd);
    phaseState = Theta0.gait.phase_i;

    out.caseName = string(caseName);
    out.tau_OC     = zeros(12, Tsim);
    out.tau_res    = zeros(12, Tsim);
    out.tau_final  = zeros(12, Tsim);
    out.fz_total   = zeros(1, Tsim);
    out.fz_target  = zeros(1, Tsim);
    out.impact     = zeros(4, Tsim);
    out.gate       = zeros(4, Tsim);
    out.slip       = zeros(4, Tsim);
    out.f_mis      = zeros(4, Tsim);
    out.track      = zeros(4, Tsim);
    out.dKp_mean   = zeros(4, Tsim);
    out.dKd_mean   = zeros(4, Tsim);
    out.torque_use = zeros(12, Tsim);
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

        [mockPosErr, mockFootVel, bodyDrift] = mismatchInjection(caseName, t, Tsim);
        impParams.mock_position_error    = mockPosErr;
        impParams.mock_foot_velocity_now = mockFootVel;

        x_hat(3) = x_hat(3) + bodyDrift(1);
        x_hat(4) = x_hat(4) + bodyDrift(2);
        x_hat(5) = x_hat(5) + bodyDrift(3);

        IMP = impedanceResidualV3(cmd.z_t, Theta, Ref, WBC, MPC, impParams);

        out.tau_OC(:,t)    = WBC.tau_OC;
        out.tau_res(:,t)   = IMP.tau_res;
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
        if isfield(IMP, 'leg')
            for leg = 1:4
                if isfield(IMP.leg(leg), 'dKp_total')
                    out.dKp_mean(leg,t) = mean(diag(IMP.leg(leg).dKp_total));
                elseif isfield(IMP.leg(leg), 'dKp')
                    out.dKp_mean(leg,t) = mean(diag(IMP.leg(leg).dKp));
                end
                if isfield(IMP.leg(leg), 'dKd_total')
                    out.dKd_mean(leg,t) = mean(diag(IMP.leg(leg).dKd_total));
                elseif isfield(IMP.leg(leg), 'dKd')
                    out.dKd_mean(leg,t) = mean(diag(IMP.leg(leg).dKd));
                end
            end
        end

        out.torque_use(:,t) = abs(IMP.tau_final) ./ max(abs(wbcParams.tau_max), 1e-9);
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
    out.metric.maxTorqueUse  = max(out.torque_use(:));
    out.metric.meanBodyHErr  = mean(abs(out.body_h_err));
    out.metric.meanRPError   = mean(vecnorm(out.rollpitch_err, 2, 1));
end

function [mockPosErr, mockFootVel, bodyDrift] = mismatchInjection(caseName, t, Tsim)
    ph = 2*pi*(t-1)/max(Tsim,1);

    mockPosErr = [ 0.002*sin(ph), 0.002*cos(ph), 0.002*cos(ph), 0.002*sin(ph); ...
                  -0.001*cos(ph), 0.001*sin(ph), 0.001*cos(ph),-0.001*sin(ph); ...
                   0.000,         0.000,         0.000,         0.000];
    mockFootVel = zeros(3,4);
    bodyDrift = [0;0;0];

    switch lower(string(caseName))
        case "nominal"
            return
        case "sponge"
            mockPosErr(3,:) = [-0.010, -0.007, -0.012, -0.008] ...
                            + 0.002*[sin(ph), cos(ph), sin(2*ph), cos(2*ph)];
            mockFootVel(3,:) = [-0.020, -0.015, -0.025, -0.018];
            bodyDrift = [-0.0025; 0.003*sin(ph); 0.002*cos(ph)];
        case "slippery"
            mockPosErr(1,:) = [ 0.006, -0.005, 0.004, -0.004];
            mockPosErr(2,:) = [-0.003, 0.004, -0.004, 0.003];
            mockFootVel(1,:) = [ 0.050, -0.040, 0.060, -0.050];
            mockFootVel(2,:) = [-0.030,  0.035,-0.040,  0.030];
            bodyDrift = [0.000; 0.004*cos(ph); -0.004*sin(ph)];
        case "mud_slope"
            mockPosErr(1,:) = [ 0.008, -0.006, 0.007, -0.005];
            mockPosErr(2,:) = [-0.004,  0.005,-0.005,  0.004];
            mockPosErr(3,:) = [-0.012, -0.010,-0.014, -0.011];
            mockFootVel(1,:) = [ 0.040, -0.035, 0.045, -0.040];
            mockFootVel(2,:) = [-0.025,  0.028,-0.030,  0.027];
            mockFootVel(3,:) = [-0.020, -0.018,-0.025, -0.022];
            bodyDrift = [-0.0030; 0.010*cos(ph); 0.014 + 0.006*sin(ph)];
        otherwise
            error("Unknown mismatch case: %s", caseName);
    end
end

function printCaseSummary(results, cases)
    fprintf('\n================ Contact Mismatch Sensitivity Summary ================\n');
    fprintf('%-12s | %10s %10s %10s %10s %10s %10s %10s\n', ...
        'case', 'tauEff', 'tauRate', 'impact', 'slip', 'fMis', 'hErr', 'tauUse');
    fprintf('%s\n', repmat('-',1,95));
    for i = 1:numel(cases)
        c = results.(cases(i)).metric;
        fprintf('%-12s | %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3f\n', ...
            char(cases(i)), c.torqueEffort, c.torqueRate, c.touchImpact, ...
            c.slipTotal, c.forceMismatch, c.meanBodyHErr, c.maxTorqueUse);
    end
end

function plotCaseMetrics(results, cases)
    n = numel(cases);
    metricNames = categorical({'torque effort','torque-rate','impact','slip','force mismatch','body height err'});
    vals = zeros(numel(metricNames), n);
    for i = 1:n
        c = results.(cases(i)).metric;
        vals(:,i) = [c.torqueEffort; c.torqueRate; c.touchImpact; c.slipTotal; c.forceMismatch; c.meanBodyHErr];
    end
    figure('Name','Mismatch metrics');
    bar(metricNames, vals);
    ylabel('metric');
    legend(cellstr(cases), 'Location','bestoutside');
    title('Contact-mismatch sensitivity metrics');
    grid on;
end

function plotCaseSignals(results, cases)
    n = numel(cases);
    figure('Name','Mismatch signal summary');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    nexttile;
    means = zeros(4,n);
    for i = 1:n, means(:,i) = mean(results.(cases(i)).impact, 2); end
    bar(means'); title('Mean impact signal'); xlabel('case'); ylabel('signal');
    set(gca,'XTickLabel',cellstr(cases)); legend('leg1','leg2','leg3','leg4','Location','best');

    nexttile;
    means = zeros(4,n);
    for i = 1:n, means(:,i) = mean(results.(cases(i)).slip, 2); end
    bar(means'); title('Mean slip signal'); xlabel('case'); ylabel('signal');
    set(gca,'XTickLabel',cellstr(cases)); legend('leg1','leg2','leg3','leg4','Location','best');

    nexttile;
    means = zeros(4,n);
    for i = 1:n, means(:,i) = mean(results.(cases(i)).f_mis, 2); end
    bar(means'); title('Mean force-mismatch signal'); xlabel('case'); ylabel('signal');
    set(gca,'XTickLabel',cellstr(cases)); legend('leg1','leg2','leg3','leg4','Location','best');

    nexttile;
    means = zeros(4,n);
    for i = 1:n, means(:,i) = mean(results.(cases(i)).track, 2); end
    bar(means'); title('Mean track-error signal'); xlabel('case'); ylabel('signal');
    set(gca,'XTickLabel',cellstr(cases)); legend('leg1','leg2','leg3','leg4','Location','best');
end

function plotCaseTorques(results, cases)
    figure('Name','Torque comparison');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    for i = 1:min(4, numel(cases))
        nexttile;
        tauNorm = vecnorm(results.(cases(i)).tau_final, 2, 1);
        plot(tauNorm, 'LineWidth', 1.5); hold on;
        dtauNorm = [0, vecnorm(diff(results.(cases(i)).tau_final, 1, 2), 2, 1)];
        plot(dtauNorm, 'LineWidth', 1.2);
        xlabel('step'); ylabel('norm');
        legend('||tau||','||Delta tau||', 'Location','best');
        title(sprintf('%s torque response', cases(i)));
        grid on;
    end
end

function plotCaseSupport(results, cases)
    figure('Name','Support and body response');
    tiledlayout(3,1,'Padding','compact','TileSpacing','compact');

    nexttile; hold on;
    for i = 1:numel(cases)
        plot(results.(cases(i)).fz_total, 'LineWidth', 1.4);
    end
    yline(results.(cases(1)).fz_target(1), '--', 'mg');
    ylabel('\Sigma f_z [N]');
    legend([cellstr(cases), {'mg'}], 'Location','best');
    title('Total vertical support force');
    grid on;

    nexttile; hold on;
    for i = 1:numel(cases)
        plot(results.(cases(i)).body_h_err, 'LineWidth', 1.4);
    end
    ylabel('h body err [m]');
    legend(cellstr(cases), 'Location','best');
    title('Body-height error');
    grid on;

    nexttile; hold on;
    for i = 1:numel(cases)
        plot(vecnorm(results.(cases(i)).rollpitch_err, 2, 1), 'LineWidth', 1.4);
    end
    ylabel('RP err norm'); xlabel('step');
    legend(cellstr(cases), 'Location','best');
    title('Roll/pitch error norm');
    grid on;
end

function plotCaseGainAdaptation(results, cases)
    n = numel(cases);
    meanDKp = zeros(4,n);
    meanDKd = zeros(4,n);
    for i = 1:n
        meanDKp(:,i) = mean(results.(cases(i)).dKp_mean, 2);
        meanDKd(:,i) = mean(results.(cases(i)).dKd_mean, 2);
    end

    figure('Name','Gain adaptation summary');
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    nexttile;
    bar(meanDKp');
    xlabel('case'); ylabel('mean diag \DeltaK_p');
    set(gca,'XTickLabel',cellstr(cases));
    legend('leg1','leg2','leg3','leg4', 'Location','best');
    title('Stiffness correction summary');
    grid on;

    nexttile;
    bar(meanDKd');
    xlabel('case'); ylabel('mean diag \DeltaK_d');
    set(gca,'XTickLabel',cellstr(cases));
    legend('leg1','leg2','leg3','leg4', 'Location','best');
    title('Damping correction summary');
    grid on;
end
