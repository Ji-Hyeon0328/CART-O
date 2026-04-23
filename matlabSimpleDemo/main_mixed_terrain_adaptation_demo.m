%% main_mixed_terrain_adaptation_demo.m
clear; clc; close all;

robotName = "go1";
Tsim = 40;
terrainTimeline = strings(1,Tsim);
terrainTimeline(1:8)   = "flat";
terrainTimeline(9:16)  = "rough";
terrainTimeline(17:22) = "flat";
terrainTimeline(23:30) = "up_slope";
terrainTimeline(31:40) = "down_slope";

[params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd] = initBaseParams();
[params0, mpcParams0, wbcParams0, impParams0, x0, meta] = ...
    applyRobotPresetV3(robotName, params0, mpcParams0, wbcParams0, impParams0, x0);

baseline = runMixedSequence("baseline", terrainTimeline, false, ...
    params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, meta);
proposed = runMixedSequence("proposed", terrainTimeline, true, ...
    params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, meta);

printMetricSummary(baseline.metrics, proposed.metrics);
plotTerrainAndCommands(terrainTimeline, baseline, proposed);
plotContactSchedules(baseline, proposed);
plotMixedTerrainMetrics(baseline, proposed);
plotTimeSeriesComparison(baseline, proposed);
disp("Done. Variables: baseline, proposed");

function [params, mpcParams, wbcParams, impParams, x_hat, u_cmd] = initBaseParams()
x_hat = [0;0;0.42; 0.03;-0.02;0.10; 0.25;0.02;0; 0;0;0.05];
u_cmd = [0.30;0.00;0.10];
params.dt = 0.02; params.N = 20;
params.robot.mass = 25; params.robot.g = 9.81; params.robot.Ibody = diag([0.45,1.20,1.30]);
params.hip_offset_body = [0.28,0.28,-0.28,-0.28; 0.16,-0.16,0.16,-0.16; 0,0,0,0];
params.p_foot_now = [0.30,0.30,-0.28,-0.28; 0.18,-0.18,0.18,-0.18; 0,0,0,0];
params.Kv_foot = diag([-0.08,-0.08,0]);

mpcParams = struct(); mpcParams.dt=params.dt; mpcParams.N=params.N; mpcParams.robot=params.robot;
mpcParams.fz_max_per_leg = 1.2*params.robot.mass*params.robot.g;
mpcParams.use_force_rate_bound = false; mpcParams.df_max = 500;
mpcParams.rho_f=0.60; mpcParams.rho_df=0.40;
mpcParams.q0.x=0; mpcParams.q0.y=0; mpcParams.q0.z=80;
mpcParams.q0.roll=80; mpcParams.q0.pitch=80; mpcParams.q0.yaw=1;
mpcParams.q0.vx=40; mpcParams.q0.vy=40; mpcParams.q0.vz=30;
mpcParams.q0.wx=1; mpcParams.q0.wy=1; mpcParams.q0.wz=30;
mpcParams.weightRange.wh=[0.5,3.0]; mpcParams.weightRange.wv=[0.5,3.0]; mpcParams.weightRange.we=[0.1,5.0];
mpcParams.H_reg=1e-8;

wbcParams = struct(); wbcParams.dt=params.dt; wbcParams.robot=params.robot;
wbcParams.fz_max_per_leg=mpcParams.fz_max_per_leg; wbcParams.mu=[];
wbcParams.tau_min=-90*ones(12,1); wbcParams.tau_max=90*ones(12,1); wbcParams.tau_prev=zeros(12,1);
wbcParams.Kp_base=diag([40,40,80,80,80,30]); wbcParams.Kd_base=diag([10,10,20,20,20,8]);
wbcParams.Kp_foot=diag([80,80,120]); wbcParams.Kd_foot=diag([10,10,15]);
wbcParams.Wb=diag([20,20,100,100,100,20]); wbcParams.Wfoot=diag([50,50,100]);
wbcParams.Wforce_per_foot=diag([1,1,3]); wbcParams.Wtau=1e-3*eye(12); wbcParams.Wdtau=1e-2*eye(12); wbcParams.H_reg=1e-8;

impParams = struct(); impParams.lambda_res=0.20; impParams.phase_width=0.10; impParams.touchdown_phase=1.0;
impParams.alpha_imp=25; impParams.beta_imp=20; impParams.s_imp0=0.25;
impParams.alpha_F=4; impParams.beta_F=3; impParams.F0=60;
impParams.alpha_slip=10; impParams.beta_slip=12; impParams.slip0=0.10;
impParams.alpha_track=15; impParams.e0=0.03;
impParams.Kp.cons.st=diag([25,25,45]); impParams.Kd.cons.st=diag([18,18,30]);
impParams.Kp.cons.sw=diag([60,60,90]); impParams.Kd.cons.sw=diag([7,7,10]);
impParams.Kp.agg.st=diag([40,40,70]); impParams.Kd.agg.st=diag([20,20,35]);
impParams.Kp.agg.sw=diag([90,90,130]); impParams.Kd.agg.sw=diag([10,10,15]);
impParams.Kp_min=diag([5,5,5]); impParams.Kp_max=diag([200,200,250]);
impParams.Kd_min=diag([1,1,1]); impParams.Kd_max=diag([80,80,100]);
impParams.mock_position_error=zeros(3,4); impParams.mock_foot_velocity_now=zeros(3,4);
end

function out = runMixedSequence(modeName, terrainTimeline, useResidual, params0, mpcParams0, wbcParams0, impParams0, x0, u_cmd, meta)
Tsim = numel(terrainTimeline);
tau_final = zeros(12,Tsim); tau_OC = zeros(12,Tsim); tau_res = zeros(12,Tsim);
fz_total = zeros(1,Tsim); force_spike = zeros(1,Tsim); impact = zeros(4,Tsim); gate = zeros(4,Tsim);
force_mismatch = zeros(4,Tsim); track_error = zeros(4,Tsim); kdes = zeros(1,Tsim); duty = zeros(4,Tsim);
hswing = zeros(4,Tsim); body_h = zeros(1,Tsim); beta = zeros(3,Tsim); contact_seq = cell(1,Tsim);
x_hat = x0;
flatCmd = terrainCmd("flat");
Theta0 = thetaDecoder(flatCmd.z_t, flatCmd.a_HL, x_hat, u_cmd); phaseState = Theta0.gait.phase_i;
for t=1:Tsim
    params=params0; mpcParams=mpcParams0; wbcParams=wbcParams0; impParams=impParams0;
    terrain = terrainTimeline(t);
    if modeName=="baseline", cmd=terrainCmd("baseline_flat"); else, cmd=terrainCmd(terrain); end
    [mockPosErr,mockFootVel] = terrainDisturbance(terrain, t, Tsim);
    impParams.mock_position_error = mockPosErr; impParams.mock_foot_velocity_now = mockFootVel;
    Theta = thetaDecoder(cmd.z_t, cmd.a_HL, x_hat, u_cmd);
    phaseState = mod(phaseState + params.dt/max(Theta.gait.T,1e-6), 1.0); Theta.gait.phase_i = phaseState;
    Ref = thetaRefMapper(Theta, x_hat, u_cmd, params);
    MPC = forceMPC(x_hat, Ref, Theta, cmd.beta_t, mpcParams);
    WBC = wbcQP_robotScaledMock(x_hat, zeros(12,1), zeros(12,1), Ref, Theta, MPC, wbcParams);
    IMP = impedanceResidualV3(cmd.z_t, Theta, Ref, WBC, MPC, impParams);
    tau_OC(:,t)=WBC.tau_OC;
    if useResidual, tau_final(:,t)=IMP.tau_final; tau_res(:,t)=IMP.tau_res; else, tau_final(:,t)=WBC.tau_OC; end
    fz_total(t)=sum(MPC.F_by_leg_z(:,1));
    dF = diff(MPC.F_by_leg_z,1,2); force_spike(t)=sum(dF(:).^2)/max(meta.mg^2,1e-9);
    impact(:,t)=IMP.impact_signal; gate(:,t)=IMP.phase_gate; force_mismatch(:,t)=IMP.force_mismatch;
    if isfield(IMP,"track_error"), track_error(:,t)=IMP.track_error; end
    kdes(t)=Theta.ctrl.k_des; duty(:,t)=Theta.gait.duty_i; hswing(:,t)=Theta.foot.h_swing_i; body_h(t)=Theta.base.h_body_ref;
    beta(:,t)=cmd.beta_t; contact_seq{t}=Ref.S;
    x_hat(1)=x_hat(1)+params.dt*u_cmd(1); x_hat(2)=x_hat(2)+params.dt*u_cmd(2); x_hat(6)=x_hat(6)+params.dt*u_cmd(3);
end
out.modeName=modeName; out.tau_final=tau_final; out.tau_OC=tau_OC; out.tau_res=tau_res; out.fz_total=fz_total;
out.force_spike=force_spike; out.impact=impact; out.gate=gate; out.force_mismatch=force_mismatch; out.track_error=track_error;
out.kdes=kdes; out.duty=duty; out.hswing=hswing; out.body_h=body_h; out.beta=beta; out.contact_seq=contact_seq;
out.metrics = mixedMetrics(out, meta);
end

function cmd = terrainCmd(terrain)
terrain = lower(string(terrain));
switch terrain
    case "baseline_flat"
        cmd.z_t=0; cmd.a_HL=[-0.1;0.0;-0.2;-0.2]; cmd.beta_t=[0.25;0.50;0.25];
    case "flat"
        cmd.z_t=1; cmd.a_HL=[0.0;0.0;-0.1;-0.1]; cmd.beta_t=[0.25;0.45;0.30];
    case "rough"
        cmd.z_t=0; cmd.a_HL=[0.7;0.1;0.6;0.3]; cmd.beta_t=[0.45;0.20;0.35];
    case "up_slope"
        cmd.z_t=0; cmd.a_HL=[0.4;-0.1;0.5;0.2]; cmd.beta_t=[0.40;0.25;0.35];
    case "down_slope"
        cmd.z_t=0; cmd.a_HL=[0.2;-0.2;0.6;0.5]; cmd.beta_t=[0.35;0.20;0.45];
end
end

function [posErr, footVel] = terrainDisturbance(terrain, t, Tsim)
terrain = lower(string(terrain)); ph = 2*pi*(t-1)/max(Tsim,1);
posErr=[0.010,-0.008,0.006,0.004; -0.004,0.004,0.006,-0.006; 0,0,-0.004,-0.003];
footVel=[0.010,0,0.010,-0.010; 0,0.008,-0.008,0; -0.030,-0.020,-0.080,-0.050];
switch terrain
    case "flat", scaleImp=0.8; scaleSlip=0.8;
    case "rough"
        scaleImp=2.5; scaleSlip=2.0;
        posErr = posErr + [0.006*sin(ph),0.004*cos(ph),0.006*cos(ph),0; 0.004*cos(ph),0.004*sin(ph),0,0.005*sin(ph); 0.010,-0.008,0.012,-0.010];
    case "up_slope"
        scaleImp=1.3; scaleSlip=1.3; posErr(3,:)=posErr(3,:)+[0.004,0.004,-0.003,-0.003]; footVel(1,:)=footVel(1,:)+[0.01,0.01,0,0];
    case "down_slope"
        scaleImp=2.2; scaleSlip=1.8; footVel(3,:)=footVel(3,:)+[-0.05,-0.03,-0.05,-0.04];
    otherwise, scaleImp=1.0; scaleSlip=1.0;
end
footVel(1:2,:)=scaleSlip*footVel(1:2,:); footVel(3,:)=scaleImp*footVel(3,:);
end

function m = mixedMetrics(out, meta)
tauNorm=vecnorm(out.tau_final,2,1); dtauNorm=vecnorm(diff(out.tau_final,1,2),2,1);
m.mean_fz_ratio = mean(out.fz_total/max(meta.mg,1e-9));
m.torque_effort = sum(tauNorm.^2); m.torque_rate = sum(dtauNorm.^2); m.force_spike = sum(out.force_spike);
m.touchdown_impact = sum((out.gate.*out.impact).^2,'all'); m.tracking_error = sum(out.track_error.^2,'all');
end

function printMetricSummary(base, ours)
fprintf("\n===== Mixed Terrain Metric Summary =====\n");
fprintf("mean fz/mg:       base %.3f | ours %.3f\n", base.mean_fz_ratio, ours.mean_fz_ratio);
fprintf("torque effort:    base %.3e | ours %.3e | ratio %.3f\n", base.torque_effort, ours.torque_effort, sr(ours.torque_effort, base.torque_effort));
fprintf("torque-rate:      base %.3e | ours %.3e | ratio %.3f\n", base.torque_rate, ours.torque_rate, sr(ours.torque_rate, base.torque_rate));
fprintf("force spike:      base %.3e | ours %.3e | ratio %.3f\n", base.force_spike, ours.force_spike, sr(ours.force_spike, base.force_spike));
fprintf("touchdown impact: base %.3e | ours %.3e | ratio %.3f\n", base.touchdown_impact, ours.touchdown_impact, sr(ours.touchdown_impact, base.touchdown_impact));
fprintf("tracking error:   base %.3e | ours %.3e | ratio %.3f\n", base.tracking_error, ours.tracking_error, sr(ours.tracking_error, base.tracking_error));
end

function r = sr(a,b), r=a/max(b,1e-12); end

function plotTerrainAndCommands(terrainTimeline, baseline, proposed)
Tsim=numel(terrainTimeline); terr=zeros(1,Tsim);
for t=1:Tsim, terr(t)=terrIdx(terrainTimeline(t)); end
figure; stairs(1:Tsim,terr,'LineWidth',1.5); yticks(1:4); yticklabels(["flat","rough","up","down"]); xlabel("step"); ylabel("terrain"); title("Mixed terrain timeline"); grid on;
figure;
subplot(4,1,1); plot(baseline.hswing(1,:),'LineWidth',1.5); hold on; plot(proposed.hswing(1,:),'LineWidth',1.5); ylabel("h swing"); legend("base","ours"); title("Decoded commands over time"); grid on;
subplot(4,1,2); plot(mean(baseline.duty,1),'LineWidth',1.5); hold on; plot(mean(proposed.duty,1),'LineWidth',1.5); ylabel("duty"); grid on;
subplot(4,1,3); plot(baseline.kdes,'LineWidth',1.5); hold on; plot(proposed.kdes,'LineWidth',1.5); ylabel("k des"); grid on;
subplot(4,1,4); plot(proposed.beta(1,:),'LineWidth',1.2); hold on; plot(proposed.beta(2,:),'LineWidth',1.2); plot(proposed.beta(3,:),'LineWidth',1.2); ylabel("beta"); xlabel("step"); legend("beta h","beta v","beta e"); grid on;
end

function idx = terrIdx(s)
s=lower(string(s));
switch s
    case "flat", idx=1;
    case "rough", idx=2;
    case "up_slope", idx=3;
    case "down_slope", idx=4;
    otherwise, idx=1;
end
end

function plotContactSchedules(baseline, proposed)
picks=[8,20,36];
figure;
for i=1:numel(picks)
    t=picks(i);
    subplot(2,numel(picks),i); imagesc(baseline.contact_seq{t}); colormap(gray); title("Baseline S_t @ t="+t); xlabel("k"); ylabel("leg"); colorbar;
    subplot(2,numel(picks),i+numel(picks)); imagesc(proposed.contact_seq{t}); colormap(gray); title("Proposed S_t @ t="+t); xlabel("k"); ylabel("leg"); colorbar;
end
end

function plotMixedTerrainMetrics(baseline, proposed)
names=["torque effort","torque-rate","force spike","touchdown impact","tracking error"];
vals=[baseline.metrics.torque_effort,proposed.metrics.torque_effort; baseline.metrics.torque_rate,proposed.metrics.torque_rate; baseline.metrics.force_spike,proposed.metrics.force_spike; baseline.metrics.touchdown_impact,proposed.metrics.touchdown_impact; baseline.metrics.tracking_error,proposed.metrics.tracking_error];
figure; bar(categorical(names), vals); ylabel("metric"); legend("baseline","proposed"); title("Mixed-terrain metric comparison"); grid on;
figure; ratio=vals(:,2)./max(vals(:,1),1e-12); bar(categorical(names), ratio); yline(1.0,"--","baseline"); ylabel("Ours / Baseline"); title("Mixed-terrain metric ratio"); grid on;
end

function plotTimeSeriesComparison(baseline, proposed)
tauNormBase=vecnorm(baseline.tau_final,2,1); tauNormOurs=vecnorm(proposed.tau_final,2,1);
dtauBase=vecnorm(diff(baseline.tau_final,1,2),2,1); dtauOurs=vecnorm(diff(proposed.tau_final,1,2),2,1);
impactBase=sum((baseline.gate.*baseline.impact).^2,1); impactOurs=sum((proposed.gate.*proposed.impact).^2,1);
figure;
subplot(3,1,1); plot(tauNormBase,'LineWidth',1.5); hold on; plot(tauNormOurs,'LineWidth',1.5); ylabel("||tau||"); legend("baseline","proposed"); title("Mixed-terrain time-series"); grid on;
subplot(3,1,2); plot(1:numel(dtauBase),dtauBase,'LineWidth',1.5); hold on; plot(1:numel(dtauOurs),dtauOurs,'LineWidth',1.5); ylabel("||Delta tau||"); grid on;
subplot(3,1,3); plot(impactBase,'LineWidth',1.5); hold on; plot(impactOurs,'LineWidth',1.5); ylabel("impact"); xlabel("step"); grid on;
end
