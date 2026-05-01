%% main_terrain_xyz_mismatch_real_pipeline_demo.m
% Terrain x mismatch-type x mismatch-level demo using the CURRENT low-level
% controller pipeline:
%   thetaDecoder -> thetaRefMapper -> forceMPC -> wbcQP_robotScaledMock
%   -> impedanceResidualV3
%
% Terrain cases:
%   solid_flat, solid_uneven, solid_up_slope, solid_down_slope
%   sponge_flat, sponge_uneven, sponge_up_slope, sponge_down_slope
%
% Scenarios:
%   ideal  : no extra contact mismatch
%   xy     : tangential foothold/contact mismatch only
%   z      : vertical sinkage/support mismatch only
%   xyz    : tangential + vertical mismatch
%
% Required files on MATLAB path:
%   applyRobotPresetV3.m, thetaDecoder.m, thetaRefMapper.m, forceMPC.m,
%   wbcQP_robotScaledMock.m, impedanceResidualV3.m

clear; clc; close all;

%% User options
robotName = "go1";             % "go1" or "spot"
Tsim      = 30;
levels    = 0.25:0.25:2.00;
scenarioList = ["ideal", "xy", "z", "xyz"];
showSummary = true;

%% Base low-level initialization
[params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd] = initBaseParams();
[params0, mpcParams0, wbcParams0, impParams0, x0, meta] = ...
    applyRobotPresetV3(robotName, params0, mpcParams0, wbcParams0, impParams0, x0);

%% Assumed high-level command
cmd.z_t    = 0;
cmd.a_HL   = [0.35; 0.10; 0.45; 0.25];
cmd.beta_t = [0.40; 0.25; 0.35];

%% Terrain grid
terrains = buildTerrains();

%% Run all experiments
results = struct();
tableRows = {};

for it = 1:numel(terrains)
    terr = terrains(it);
    for is = 1:numel(scenarioList)
        scenario = scenarioList(is);
        if scenario == "ideal"
            levelSet = 0;
        else
            levelSet = levels;
        end
        for il = 1:numel(levelSet)
            lvl = levelSet(il);
            out = runRealPipelineCase(terr, scenario, lvl, cmd, ...
                params0, mpcParams0, wbcParams0, impParams0, ...
                x0, u_cmd, Tsim, meta);
            key = makeResultKey(terr.name, scenario, lvl);
            results.(key) = out;
            tableRows(end+1,:) = {char(terr.name), char(terr.compliance), ...
                char(terr.geometry), char(scenario), lvl, ...
                out.metric.torqueEffort, out.metric.torqueRate, ...
                out.metric.touchImpact, out.metric.slipTotal, ...
                out.metric.forceMismatch, out.metric.trackError, ...
                out.metric.meanBodyHErr, out.metric.meanRPError, ...
                out.metric.maxTorqueUse}; %#ok<AGROW>
        end
    end
end

summaryTable = cell2table(tableRows, 'VariableNames', { ...
    'terrain','compliance','geometry','scenario','level', ...
    'torqueEffort','torqueRate','touchImpact','slipTotal', ...
    'forceMismatch','trackError','meanBodyHErr','meanRPError','maxTorqueUse'});
summaryTable = addNominalRatios(summaryTable, terrains);

if showSummary
    disp(" ");
    disp("================ First rows of summaryTable ================");
    disp(summaryTable(1:min(20,height(summaryTable)),:));
end

%% Plots
plotGridMetricOverview(summaryTable, terrains, scenarioList);
plotMismatchLevelSweep(summaryTable, terrains);
plotRepresentativeTerrainBars(summaryTable, terrains);
plotRepresentativeTimeSeries(results, terrains);
plotTerrainMismatchDiagnostics(results, terrains, levels);

disp("Done.");
disp("Main outputs: results, summaryTable");

%% ========================================================================
%% Local functions
%% ========================================================================

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
    mpcParams.q0.x=0; mpcParams.q0.y=0; mpcParams.q0.z=80;
    mpcParams.q0.roll=80; mpcParams.q0.pitch=80; mpcParams.q0.yaw=1;
    mpcParams.q0.vx=40; mpcParams.q0.vy=40; mpcParams.q0.vz=30;
    mpcParams.q0.wx=1; mpcParams.q0.wy=1; mpcParams.q0.wz=30;
    mpcParams.weightRange.wh=[0.5,3.0];
    mpcParams.weightRange.wv=[0.5,3.0];
    mpcParams.weightRange.we=[0.1,5.0];
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
    impParams.alpha_imp=25; impParams.beta_imp=20; impParams.s_imp0=0.25;
    impParams.alpha_F=4; impParams.beta_F=3; impParams.F0=60;
    impParams.alpha_slip=10; impParams.beta_slip=12; impParams.slip0=0.10;
    impParams.alpha_track=15; impParams.e0=0.03;
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

function terrains = buildTerrains()
    names = ["solid_flat","solid_uneven","solid_up_slope","solid_down_slope", ...
             "sponge_flat","sponge_uneven","sponge_up_slope","sponge_down_slope"];
    compliance = ["solid","solid","solid","solid", "sponge","sponge","sponge","sponge"];
    geometry = ["flat","uneven","up","down", "flat","uneven","up","down"];
    xySens   = [1.00, 1.18, 1.08, 1.08, 1.05, 1.25, 1.18, 1.18];
    zSens    = [0.60, 0.95, 1.05, 1.05, 1.45, 1.80, 1.75, 1.75];
    poseSens = [1.00, 1.20, 1.35, 1.35, 1.05, 1.30, 1.45, 1.45];

    template = struct('name',"",'compliance',"",'geometry',"", ...
        'xySens',1.0,'zSens',1.0,'poseSens',1.0, ...
        'basePitch',0.0,'baseRoll',0.0,'bodyHeightBias',0.0,'terrainWeightScale',1.0);
    terrains = repmat(template, 1, numel(names));

    for i = 1:numel(names)
        terr = template;
        terr.name = names(i); terr.compliance = compliance(i); terr.geometry = geometry(i);
        terr.xySens = xySens(i); terr.zSens = zSens(i); terr.poseSens = poseSens(i);

        switch geometry(i)
            case "flat"
                terr.basePitch=0.00; terr.baseRoll=0.00; terr.bodyHeightBias=0.00; terr.terrainWeightScale=1.00;
            case "uneven"
                terr.basePitch=0.02; terr.baseRoll=0.02; terr.bodyHeightBias=0.015; terr.terrainWeightScale=1.10;
            case "up"
                terr.basePitch=0.08; terr.baseRoll=0.00; terr.bodyHeightBias=0.010; terr.terrainWeightScale=1.08;
            case "down"
                terr.basePitch=-0.08; terr.baseRoll=0.00; terr.bodyHeightBias=0.005; terr.terrainWeightScale=1.08;
        end
        if compliance(i) == "sponge"
            terr.bodyHeightBias = terr.bodyHeightBias - 0.010;
            terr.terrainWeightScale = terr.terrainWeightScale * 1.08;
        end
        terrains(i) = terr;
    end
end

function out = runRealPipelineCase(terr, scenario, level, cmd, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, Tsim, meta)
    x_hat = applyTerrainToInitialState(x0, terr);
    cmdLocal = applyTerrainToHLCommand(cmd, terr, scenario);
    Theta0 = thetaDecoder(cmdLocal.z_t, cmdLocal.a_HL, x_hat, u_cmd);
    phaseState = Theta0.gait.phase_i;

    out.terrainName = terr.name; out.scenario = scenario; out.level = level;
    out.tau_OC=zeros(12,Tsim); out.tau_res=zeros(12,Tsim); out.tau_final=zeros(12,Tsim);
    out.fz_total=zeros(1,Tsim); out.fz_target=zeros(1,Tsim);
    out.impact=zeros(4,Tsim); out.gate=zeros(4,Tsim); out.slip=zeros(4,Tsim);
    out.f_mis=zeros(4,Tsim); out.track=zeros(4,Tsim);
    out.body_h_err=zeros(1,Tsim); out.rollpitch_err=zeros(2,Tsim);
    out.torque_use=zeros(12,Tsim); out.dKp_mean=zeros(4,Tsim); out.dKd_mean=zeros(4,Tsim);

    for t = 1:Tsim
        params = params0;
        mpcParams = applyTerrainToMPCParams(mpcParams0, terr);
        wbcParams = applyTerrainToWBCParams(wbcParams0, terr);
        impParams = impParams0;

        Theta = thetaDecoder(cmdLocal.z_t, cmdLocal.a_HL, x_hat, u_cmd);
        Theta = applyTerrainToTheta(Theta, terr);
        phaseState = mod(phaseState + params.dt / max(Theta.gait.T, 1e-6), 1.0);
        Theta.gait.phase_i = phaseState;
        Ref = thetaRefMapper(Theta, x_hat, u_cmd, params);
        MPC = forceMPC(x_hat, Ref, Theta, cmdLocal.beta_t, mpcParams);
        WBC = wbcQP_robotScaledMock(x_hat, zeros(12,1), zeros(12,1), Ref, Theta, MPC, wbcParams);

        [mockPosErr, mockFootVel, bodyDrift] = terrainMismatchInjection(terr, scenario, level, t, Tsim);
        impParams.mock_position_error = mockPosErr;
        impParams.mock_foot_velocity_now = mockFootVel;

        x_hat(3) = x_hat(3) + bodyDrift(1);
        x_hat(4) = x_hat(4) + bodyDrift(2);
        x_hat(5) = x_hat(5) + bodyDrift(3);

        IMP = impedanceResidualV3(cmdLocal.z_t, Theta, Ref, WBC, MPC, impParams);

        out.tau_OC(:,t)    = WBC.tau_OC;
        out.tau_res(:,t)   = getFieldOrZero(IMP, 'tau_res', zeros(12,1));
        out.tau_final(:,t) = IMP.tau_final;
        out.fz_total(t)  = sum(MPC.F_by_leg_z(:,1)); out.fz_target(t) = meta.mg;
        out.impact(:,t) = getFieldOrZero(IMP, 'impact_signal', zeros(4,1));
        out.gate(:,t)   = getFieldOrZero(IMP, 'phase_gate', zeros(4,1));
        out.slip(:,t)   = getFieldOrZeroMulti(IMP, ["slip_signal","slip"], zeros(4,1));
        out.f_mis(:,t)  = getFieldOrZeroMulti(IMP, ["force_mismatch","force_mismatch_norm"], zeros(4,1));
        out.track(:,t)  = getFieldOrZeroMulti(IMP, ["track_error","tracking_error"], zeros(4,1));

        if isfield(IMP, 'leg')
            for leg = 1:4
                out.dKp_mean(leg,t) = getLegDiagMean(IMP.leg(leg), ["dKp_total","dKp","dKp_force"]);
                out.dKd_mean(leg,t) = getLegDiagMean(IMP.leg(leg), ["dKd_total","dKd","dKd_force"]);
            end
        end

        out.torque_use(:,t) = abs(IMP.tau_final) ./ max(abs(wbcParams.tau_max), 1e-9);
        out.body_h_err(t) = getThetaBodyHeightRef(Theta, x_hat) - x_hat(3);
        out.rollpitch_err(:,t) = [getThetaRollRef(Theta) - x_hat(4); getThetaPitchRef(Theta) - x_hat(5)];

        x_hat(1) = x_hat(1) + params.dt * u_cmd(1);
        x_hat(2) = x_hat(2) + params.dt * u_cmd(2);
        x_hat(6) = x_hat(6) + params.dt * u_cmd(3);
        x_hat(3) = max(0.15, min(0.70, x_hat(3)));
        x_hat(4) = max(-0.40, min(0.40, x_hat(4)));
        x_hat(5) = max(-0.40, min(0.40, x_hat(5)));
    end

    tauNorm  = vecnorm(out.tau_final,2,1);
    dtauNorm = vecnorm(diff(out.tau_final,1,2),2,1);
    out.metric.torqueEffort  = sum(tauNorm.^2);
    out.metric.torqueRate    = sum(dtauNorm.^2);
    out.metric.touchImpact   = sum((out.gate .* out.impact).^2, 'all');
    out.metric.slipTotal     = sum(out.slip.^2, 'all');
    out.metric.forceMismatch = sum(out.f_mis.^2, 'all');
    out.metric.trackError    = sum(out.track.^2, 'all');
    out.metric.supportErr    = sum((out.fz_total - out.fz_target).^2);
    out.metric.maxTorqueUse  = max(out.torque_use(:));
    out.metric.meanBodyHErr  = mean(abs(out.body_h_err));
    out.metric.meanRPError   = mean(vecnorm(out.rollpitch_err,2,1));
    out.metric.meanDKp       = mean(abs(out.dKp_mean), 'all');
    out.metric.meanDKd       = mean(abs(out.dKd_mean), 'all');
end

function x = applyTerrainToInitialState(x, terr)
    x(3)=x(3)+terr.bodyHeightBias; x(4)=x(4)+terr.baseRoll; x(5)=x(5)+terr.basePitch;
end

function cmdOut = applyTerrainToHLCommand(cmdIn, terr, scenario)
    cmdOut = cmdIn;
    if terr.compliance == "sponge"
        cmdOut.a_HL(1)=min(1.0, cmdOut.a_HL(1)+0.08); cmdOut.a_HL(4)=min(1.0, cmdOut.a_HL(4)+0.10);
    end
    if terr.geometry == "uneven"
        cmdOut.a_HL(1)=min(1.0, cmdOut.a_HL(1)+0.08); cmdOut.a_HL(3)=min(1.0, cmdOut.a_HL(3)+0.06);
    elseif terr.geometry == "up" || terr.geometry == "down"
        cmdOut.a_HL(2)=min(1.0, cmdOut.a_HL(2)+0.05); cmdOut.a_HL(3)=min(1.0, cmdOut.a_HL(3)+0.04);
    end
    if scenario == "xy"
        cmdOut.beta_t = normalizeBeta(cmdOut.beta_t + [0.00;0.08;-0.03]);
    elseif scenario == "z"
        cmdOut.beta_t = normalizeBeta(cmdOut.beta_t + [0.08;-0.02;0.00]);
    elseif scenario == "xyz"
        cmdOut.beta_t = normalizeBeta(cmdOut.beta_t + [0.05;0.05;-0.02]);
    end
end

function b = normalizeBeta(b), b=max(b,0.01); b=b/sum(b); end

function mpcParams = applyTerrainToMPCParams(mpcParams, terr)
    mpcParams.rho_f = mpcParams.rho_f * terr.terrainWeightScale;
    mpcParams.rho_df = mpcParams.rho_df * terr.terrainWeightScale;
end

function wbcParams = applyTerrainToWBCParams(wbcParams, terr)
    if terr.compliance == "sponge"
        if isempty(wbcParams.mu), wbcParams.mu = 0.55;
        else, wbcParams.mu = max(0.35, 0.85*wbcParams.mu); end
    end
end

function Theta = applyTerrainToTheta(Theta, terr)
    if isfield(Theta,'base')
        if isfield(Theta.base,'h_body_ref'), Theta.base.h_body_ref = Theta.base.h_body_ref + terr.bodyHeightBias; end
        if isfield(Theta.base,'pitch_ref'), Theta.base.pitch_ref = Theta.base.pitch_ref + terr.basePitch; end
        if isfield(Theta.base,'roll_ref'), Theta.base.roll_ref = Theta.base.roll_ref + terr.baseRoll; end
    end
end

function [mockPosErr, mockFootVel, bodyDrift] = terrainMismatchInjection(terr, scenario, level, t, Tsim)
    ph = 2*pi*(t-1)/max(Tsim,1);
    mockPosErr=zeros(3,4); mockFootVel=zeros(3,4); bodyDrift=[0;0;0];

    if terr.geometry == "uneven"
        mockPosErr(3,:) = mockPosErr(3,:) + 0.0015*[sin(ph),cos(ph),sin(2*ph),cos(2*ph)];
        bodyDrift = bodyDrift + [0;0.0015*cos(ph);0.0015*sin(ph)];
    elseif terr.geometry == "up" || terr.geometry == "down"
        bodyDrift = bodyDrift + [0;0;0.001*sign(terr.basePitch)];
    end

    if scenario == "ideal", return; end
    applyXY = (scenario == "xy") || (scenario == "xyz");
    applyZ  = (scenario == "z")  || (scenario == "xyz");

    if applyXY
        xyScale = level * terr.xySens;
        mockPosErr(1,:) = mockPosErr(1,:) + xyScale*[0.006,-0.005,0.004,-0.004];
        mockPosErr(2,:) = mockPosErr(2,:) + xyScale*[-0.003,0.004,-0.004,0.003];
        mockFootVel(1,:) = mockFootVel(1,:) + xyScale*[0.050,-0.040,0.060,-0.050];
        mockFootVel(2,:) = mockFootVel(2,:) + xyScale*[-0.030,0.035,-0.040,0.030];
        bodyDrift = bodyDrift + [0; level*0.0035*terr.poseSens*cos(ph); -level*0.0035*terr.poseSens*sin(ph)];
    end
    if applyZ
        zScale = level * terr.zSens;
        mockPosErr(3,:) = mockPosErr(3,:) + zScale*([-0.010,-0.007,-0.012,-0.008] + 0.002*[sin(ph),cos(ph),sin(2*ph),cos(2*ph)]);
        mockFootVel(3,:) = mockFootVel(3,:) + zScale*[-0.020,-0.015,-0.025,-0.018];
        bodyDrift = bodyDrift + [-level*0.0025*terr.zSens; level*0.0015*terr.poseSens*sin(ph); level*0.0015*terr.poseSens*cos(ph)];
    end
end

function key = makeResultKey(terrainName, scenario, level)
    lvl = round(level*100);
    key = matlab.lang.makeValidName(sprintf('%s__%s__L%03d', char(terrainName), char(scenario), lvl));
end

function T = addNominalRatios(T, terrains)
    metricNames = ["torqueEffort","torqueRate","touchImpact","slipTotal","forceMismatch","trackError","meanBodyHErr","meanRPError","maxTorqueUse"];
    for m = metricNames, T.(m+"_ratio") = nan(height(T),1); end
    for it = 1:numel(terrains)
        terrName = char(terrains(it).name);
        idxTerr = strcmp(T.terrain, terrName); idxNom = idxTerr & strcmp(T.scenario,'ideal');
        for m = metricNames
            denom = T.(m)(idxNom);
            if isempty(denom) || abs(denom(1)) < 1e-12, denom = 1e-12; else, denom = denom(1); end
            T.(m+"_ratio")(idxTerr) = T.(m)(idxTerr) ./ denom;
        end
    end
end

function plotGridMetricOverview(T, terrains, scenarioList)
    figure('Name','Figure 1: Terrain grid metric overview','Color','w');
    tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
    metrics = ["torqueEffort_ratio","torqueRate_ratio","touchImpact_ratio","slipTotal_ratio","meanBodyHErr_ratio","meanRPError_ratio"];
    titles = ["Torque effort","Torque-rate","Touchdown impact","Slip","Body-height error","Roll/pitch error"];
    for k = 1:numel(metrics)
        nexttile; hold on; grid on;
        for is = 2:numel(scenarioList)
            sc = scenarioList(is); means = zeros(numel(terrains),1);
            for it = 1:numel(terrains)
                idx = strcmp(T.terrain,char(terrains(it).name)) & strcmp(T.scenario,char(sc));
                means(it) = mean(T.(metrics(k))(idx),'omitnan');
            end
            plot(1:numel(terrains), means, '-o','LineWidth',1.5,'DisplayName',char(sc));
        end
        yline(1,'--','nominal'); xticks(1:numel(terrains)); xticklabels({terrains.name}); xtickangle(35);
        ylabel('ratio to terrain ideal'); title(titles(k)); legend('Location','best');
    end
end

function plotMismatchLevelSweep(T, terrains)
    figure('Name','Figure 2: Mismatch level sweep','Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    selectedTerrains = ["solid_flat","solid_uneven","sponge_flat","sponge_uneven"];
    selectedMetric = ["slipTotal_ratio","touchImpact_ratio","meanBodyHErr_ratio","meanRPError_ratio"];
    selectedTitle = ["Slip ratio","Impact ratio","Body-height ratio","Roll/pitch ratio"];
    for km = 1:numel(selectedMetric)
        nexttile; hold on; grid on;
        for terrName = selectedTerrains
            for sc = ["xy","z","xyz"]
                idx = strcmp(T.terrain,char(terrName)) & strcmp(T.scenario,char(sc));
                tt = T(idx,:); if isempty(tt), continue; end
                [lv, ord] = sort(tt.level); yy = tt.(selectedMetric(km)); yy=yy(ord);
                if sc == "xy", style='-o'; elseif sc == "z", style='--s'; else, style='-.^'; end
                plot(lv, yy, style, 'LineWidth',1.2, 'DisplayName',sprintf('%s-%s',char(terrName),char(sc)));
            end
        end
        yline(1,'k--','nominal'); xlabel('mismatch level'); ylabel('ratio to terrain ideal'); title(selectedTitle(km)); legend('Location','bestoutside');
    end
end

function plotRepresentativeTerrainBars(T, terrains)
    repLevel = 1.00;
    figure('Name','Figure 3: Representative terrain bars','Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    metrics = ["slipTotal_ratio","touchImpact_ratio","meanBodyHErr_ratio","meanRPError_ratio"];
    titles = ["Slip @ L=1","Impact @ L=1","Body-height @ L=1","Roll/pitch @ L=1"];
    scenarios = ["xy","z","xyz"];
    for km = 1:numel(metrics)
        nexttile; grid on; B = nan(numel(terrains), numel(scenarios));
        for it = 1:numel(terrains)
            for is = 1:numel(scenarios)
                idx = strcmp(T.terrain,char(terrains(it).name)) & strcmp(T.scenario,char(scenarios(is))) & abs(T.level-repLevel)<1e-12;
                if any(idx), B(it,is)=mean(T.(metrics(km))(idx),'omitnan'); end
            end
        end
        bar(B); yline(1,'--','nominal'); xticks(1:numel(terrains)); xticklabels({terrains.name}); xtickangle(35);
        ylabel('ratio to terrain ideal'); title(titles(km)); legend(cellstr(scenarios),'Location','best');
    end
end

function plotRepresentativeTimeSeries(results, terrains)
    repLevel = 1.00;
    figure('Name','Figure 4: Representative time-series','Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    pairs = {"solid_flat","xy"; "sponge_flat","z"; "solid_uneven","xyz"; "sponge_uneven","xyz"};
    for p = 1:size(pairs,1)
        terrName=pairs{p,1}; scenario=pairs{p,2};
        keyNom=makeResultKey(terrName,"ideal",0); keyRep=makeResultKey(terrName,scenario,repLevel);
        if ~isfield(results,keyNom) || ~isfield(results,keyRep), continue; end
        rNom=results.(keyNom); rRep=results.(keyRep);
        nexttile; hold on; grid on;
        yyaxis left; plot(vecnorm(rNom.tau_final,2,1),'k','LineWidth',1.5); plot(vecnorm(rRep.tau_final,2,1),'LineWidth',1.5); ylabel('||tau||');
        yyaxis right; plot(mean(rRep.slip,1),'--','LineWidth',1.2); plot(mean(rRep.impact,1),'-.','LineWidth',1.2); ylabel('mean slip / impact');
        xlabel('step'); title(sprintf('%s | %s | L=%.2f', terrName, scenario, repLevel)); legend('nom tau','mismatch tau','mean slip','mean impact','Location','best');
    end
end

function plotTerrainMismatchDiagnostics(results, terrains, levels)
    figure('Name','Figure 5: Solid vs sponge diagnostics','Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    solid = ["solid_flat","solid_uneven","solid_up_slope","solid_down_slope"];
    sponge = ["sponge_flat","sponge_uneven","sponge_up_slope","sponge_down_slope"];
    plotDiagGroup(results, solid, "xy", levels, "slip", 1); title('Solid terrains: XY mismatch -> slip');
    plotDiagGroup(results, sponge, "xy", levels, "slip", 2); title('Sponge terrains: XY mismatch -> slip');
    plotDiagGroup(results, solid, "z", levels, "height", 3); title('Solid terrains: Z mismatch -> body height');
    plotDiagGroup(results, sponge, "z", levels, "height", 4); title('Sponge terrains: Z mismatch -> body height');
end

function plotDiagGroup(results, terrainNames, scenario, levels, mode, tileIdx)
    nexttile(tileIdx); hold on; grid on;
    for terrName = terrainNames
        y = nan(size(levels)); keyNom = makeResultKey(terrName,"ideal",0);
        if ~isfield(results,keyNom), continue; end
        rNom = results.(keyNom);
        if mode == "slip", denom = sum(rNom.slip.^2,'all'); elseif mode == "height", denom = mean(abs(rNom.body_h_err)); else, denom=1; end
        denom = max(denom,1e-12);
        for i = 1:numel(levels)
            key=makeResultKey(terrName,scenario,levels(i)); if ~isfield(results,key), continue; end
            r=results.(key);
            if mode == "slip", y(i)=sum(r.slip.^2,'all')/denom; elseif mode == "height", y(i)=mean(abs(r.body_h_err))/denom; end
        end
        plot(levels,y,'-o','LineWidth',1.4,'DisplayName',char(terrName));
    end
    yline(1,'--','nominal'); xlabel('mismatch level'); ylabel('ratio to terrain ideal'); legend('Location','best');
end

function v = getFieldOrZero(S, fieldName, defaultVal)
    if isfield(S, fieldName), v = S.(fieldName); else, v = defaultVal; end
end

function v = getFieldOrZeroMulti(S, fieldNames, defaultVal)
    v = defaultVal;
    for f = fieldNames
        if isfield(S, char(f)), v = S.(char(f)); return; end
    end
end

function m = getLegDiagMean(legStruct, fieldNames)
    m = 0;
    for f = fieldNames
        if isfield(legStruct, char(f))
            A = legStruct.(char(f));
            if ismatrix(A), m = mean(diag(A)); else, m = mean(A(:)); end
            return;
        end
    end
end

function h = getThetaBodyHeightRef(Theta, x_hat)
    h = x_hat(3);
    if isfield(Theta,'base') && isfield(Theta.base,'h_body_ref'), h = Theta.base.h_body_ref; end
end
function r = getThetaRollRef(Theta)
    r=0; if isfield(Theta,'base') && isfield(Theta.base,'roll_ref'), r = Theta.base.roll_ref; end
end
function p = getThetaPitchRef(Theta)
    p=0; if isfield(Theta,'base') && isfield(Theta.base,'pitch_ref'), p = Theta.base.pitch_ref; end
end
