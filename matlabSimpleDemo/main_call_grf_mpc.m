%% main_call_grf_mpc.m
% Full prototype pipeline:
%   z_t, a_HL, x_hat, u_cmd
%       -> thetaDecoder()
%       -> thetaRefMapper()
%       -> forceMPC()
%
% State ordering:
%   x_hat = [x y z roll pitch yaw vx vy vz wx wy wz]'
%
% Leg order:
%   1 = LF, 2 = RF, 3 = LH, 4 = RH

clear; clc;

%% Example inputs
z_t = 0;  % 0 = conservative, 1 = aggressive

% a_HL = [a_swing, a_body, a_duty, a_imp]'
% normalized action, assumed in [-1, 1]
a_HL = [
    0.20;
   -0.30;
    0.40;
   -0.10
];

% beta = [beta_height, beta_velocity, beta_energy]'
beta_t = [
    0.30;
    0.40;
    0.30
];

x_hat = [
    0.00; 0.00; 0.42; ...
    0.03; -0.02; 0.10; ...
    0.25; 0.02; 0.00; ...
    0.00; 0.00; 0.05
];

u_cmd = [
    0.30;
    0.00;
    0.10
];

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

% Current foot positions in world frame.
% In a real robot, these come from FK using q_j and base pose.
params.p_foot_now = [
     0.30,  0.30, -0.28, -0.28;
     0.18, -0.18,  0.18, -0.18;
     0.00,  0.00,  0.00,  0.00
];

%% MPC-specific parameters
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
mpcParams.q0.z  = 20.0;

mpcParams.q0.roll  = 80.0;
mpcParams.q0.pitch = 80.0;
mpcParams.q0.yaw   = 1.0;

mpcParams.q0.vx = 40.0;
mpcParams.q0.vy = 40.0;
mpcParams.q0.vz = 2.0;

mpcParams.q0.wx = 1.0;
mpcParams.q0.wy = 1.0;
mpcParams.q0.wz = 30.0;

mpcParams.weightRange.wh = [0.5, 3.0];
mpcParams.weightRange.wv = [0.5, 3.0];
mpcParams.weightRange.we = [0.1, 5.0];

mpcParams.H_reg = 1e-8;

%% Run pipeline
Theta = thetaDecoder(z_t, a_HL, x_hat, u_cmd);
Ref   = thetaRefMapper(Theta, x_hat, u_cmd, params);

MPC = forceMPC(x_hat, Ref, Theta, beta_t, mpcParams);

%% Display key outputs
disp("===== Theta_HL =====");
disp(Theta);

disp("===== Contact schedule S_t: 1=stance, 0=swing =====");
disp(Ref.S);

disp("===== First optimized GRF f_t_star =====");
disp(MPC.f_t_star);

disp("===== QP exitflag =====");
disp(MPC.exitflag);
disp(MPC.message);

F_first = reshape(MPC.f_t_star, 3, 4);
disp("===== First optimized GRF by leg, columns=[LF RF LH RH] =====");
disp(F_first);

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
plot(1:params.N, Ref.Xb_ref(3,:), "LineWidth", 1.5); hold on;
plot(1:params.N, MPC.X_pred(3,:), "--", "LineWidth", 1.5);
xlabel("Horizon step");
ylabel("Base height z [m]");
legend("z ref", "z predicted by force MPC");
title("Base height reference vs predicted");
grid on;
