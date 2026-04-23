%% main_call_full_pipeline.m
% Full prototype pipeline:
%   Theta Decoder -> Theta-to-Reference Mapper -> Force MPC -> Mock WBC QP
%
% IMPORTANT:
%   This WBC is a STRUCTURAL / INTERFACE prototype.
%   It uses a simplified/mock robot model instead of URDF-derived dynamics.
%   Replace mockRobotModel() inside wbcQP_mock.m with real M,h,Jc,Jfoot later.

clear; clc;

%% Example inputs
z_t = 0;  % 0 = conservative, 1 = aggressive

a_HL = [0.20; -0.30; 0.40; -0.10];
beta_t = [0.30; 0.40; 0.30];

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
wbcParams.mu = [];  % empty -> use Theta.ctrl.mu_exp
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

%% Run pipeline
Theta = thetaDecoder(z_t, a_HL, x_hat, u_cmd);
Ref   = thetaRefMapper(Theta, x_hat, u_cmd, params);
MPC   = forceMPC(x_hat, Ref, Theta, beta_t, mpcParams);

% Mock joint state. In real robot, these come from proprioception.
qj  = zeros(12,1);
dqj = zeros(12,1);

WBC = wbcQP_mock(x_hat, qj, dqj, Ref, Theta, MPC, wbcParams);

%% Display outputs
disp("===== Force MPC: first GRF f_t_star =====");
disp(MPC.f_t_star);

disp("===== WBC: stance legs =====");
disp(WBC.stanceLegs');

disp("===== WBC: tau_OC =====");
disp(WBC.tau_OC);

disp("===== WBC: optimized contact force f_c =====");
disp(WBC.f_c);

disp("===== WBC exitflag =====");
disp(WBC.exitflag);
disp(WBC.message);

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
bar(WBC.tau_OC);
xlabel("Joint index");
ylabel("Torque [Nm]");
title("Mock WBC torque output");
grid on;
