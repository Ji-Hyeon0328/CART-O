function [params, mpcParams, wbcParams, impParams, x_hat, meta] = applyRobotPresetV3(robotName, params, mpcParams, wbcParams, impParams, x_hat)
% applyRobotPresetV3.m
%
% Robot preset helper for the CARTO low-level prototype.
%
% It updates:
%   params.robot.mass
%   params.robot.g
%   params.robot.Ibody
%   params.hip_offset_body
%   params.p_foot_now
%   mpcParams.robot
%   mpcParams.fz_max_per_leg
%   wbcParams.robot
%   wbcParams.fz_max_per_leg
%   wbcParams.tau_min / tau_max
%   wbcParams.hip_offset_body
%   wbcParams.p_foot_now
%   impParams.F0
%   impedance lookup table scale
%
% Supported presets:
%   "toy25"
%   "go1"
%   "spot"
%
% IMPORTANT:
%   This preset system makes the current prototype scale-aware.
%   It does NOT replace mock WBC with true URDF dynamics.
%   Real WBC still needs M(q), h(q,dq), J(q), FK(q) from URDF/Pinocchio/RBDL.

    robotName = lower(string(robotName));

    switch robotName
        case "toy25"
            meta.name = "toy25";
            meta.description = "Original 25 kg demo robot";

            mass = 25.0;
            g = 9.81;
            Ibody = diag([0.45, 1.20, 1.30]);

            % Project leg order: [LF, RF, LH, RH]
            hip_offset_body = [
                 0.28,  0.28, -0.28, -0.28;
                 0.16, -0.16,  0.16, -0.16;
                 0.00,  0.00,  0.00,  0.00
            ];

            nominal_body_height = 0.42;

            p_foot_now = [
                 0.30,  0.30, -0.28, -0.28;
                 0.18, -0.18,  0.18, -0.18;
                 0.00,  0.00,  0.00,  0.00
            ];

            tau_leg = [60; 60; 80];
            impScale = 1.0;

        case "go1"
            meta.name = "go1";
            meta.description = "Approximate Unitree Go1 preset";

            mass = 12.0;
            g = 9.81;

            % Approximate trunk inertia from public Go1 URDF trunk inertial.
            Ibody = [
                0.0168128557, -0.0002296769, -0.0002945293;
               -0.0002296769,  0.0630095650, -0.0000418731;
               -0.0002945293, -0.0000418731,  0.0716547275
            ];

            % Project order: [LF, RF, LH, RH]
            hip_offset_body = [
                 0.11215,  0.11215, -0.11215, -0.11215;
                 0.04675, -0.04675,  0.04675, -0.04675;
                 0.00000,  0.00000,  0.00000,  0.00000
            ];

            nominal_body_height = 0.30;

            p_foot_now = [
                 0.18,  0.18, -0.18, -0.18;
                 0.11, -0.11,  0.11, -0.11;
                 0.00,  0.00,  0.00,  0.00
            ];

            % Go1 public URDF-like effort limits: hip/thigh about 23.7, calf about 35.55 Nm.
            tau_leg = [23.7; 23.7; 35.55];

            impScale = 0.55;

        case "spot"
            meta.name = "spot";
            meta.description = "Approximate Boston Dynamics Spot preset";

            % Boston Dynamics support currently lists net mass/weight with battery as 33.8 kg.
            mass = 33.8;
            g = 9.81;

            % Approximate trunk inertia.
            % True Spot inertial parameters should come from the official/full URDF.
            % This is a box-based reduced-model placeholder for MPC/mock-WBC testing.
            Ibody = diag([0.40, 1.85, 2.10]);

            % Approximate hip geometry based on Spot-like body dimensions.
            % Project order: [LF, RF, LH, RH]
            hip_offset_body = [
                 0.33,  0.33, -0.33, -0.33;
                 0.19, -0.19,  0.19, -0.19;
                 0.00,  0.00,  0.00,  0.00
            ];

            % Spot walking min height is often in the 0.52 m range, max around 0.70 m.
            nominal_body_height = 0.55;

            p_foot_now = [
                 0.38,  0.38, -0.38, -0.38;
                 0.22, -0.22,  0.22, -0.22;
                 0.00,  0.00,  0.00,  0.00
            ];

            % Placeholder torque limits for mock WBC only.
            % Replace with actuator-specific limits if available.
            tau_leg = [80; 80; 100];

            % Heavier robot; start with moderately higher impedance than toy25.
            impScale = 1.20;

        otherwise
            error("Unknown robotName. Use 'toy25', 'go1', or 'spot'.");
    end

    %% Apply physical robot parameters
    params.robot.mass = mass;
    params.robot.g = g;
    params.robot.Ibody = Ibody;
    params.hip_offset_body = hip_offset_body;
    params.p_foot_now = p_foot_now;

    %% Update state height to match preset
    if numel(x_hat) >= 3
        x_hat(3) = nominal_body_height;
    end

    %% Force MPC dependent parameters
    mpcParams.robot = params.robot;

    % Per-leg vertical force bound.
    % 1.6*mg leaves room for transient load transfer.
    mpcParams.fz_max_per_leg = 1.6 * mass * g;

    if isfield(mpcParams, "q0")
        mpcParams.q0.z  = max(mpcParams.q0.z, 80.0);
        mpcParams.q0.vz = max(mpcParams.q0.vz, 30.0);
    end

    %% WBC dependent parameters
    wbcParams.robot = params.robot;
    wbcParams.fz_max_per_leg = mpcParams.fz_max_per_leg;

    % Geometry needed by robot-scaled mock WBC.
    wbcParams.hip_offset_body = params.hip_offset_body;
    wbcParams.p_foot_now = params.p_foot_now;

    tau_max = repmat(tau_leg, 4, 1);
    wbcParams.tau_max = tau_max;
    wbcParams.tau_min = -tau_max;
    
    %% Robot-specific WBC task weight tuning
    if robotName == "spot"
        % Heavier robot: avoid excessive torque by making force tracking softer
        % and torque regularization stronger.
        wbcParams.Wforce_per_foot = diag([0.2, 0.2, 0.8]);
        wbcParams.Wtau  = 2e-2 * eye(12);
        wbcParams.Wdtau = 1e-1 * eye(12);

    elseif robotName == "go1"
        % Smaller robot: keep force tracking moderate and torque regularization mild.
        wbcParams.Wforce_per_foot = diag([1.0, 1.0, 3.0]);
        wbcParams.Wtau  = 1e-3 * eye(12);
        wbcParams.Wdtau = 1e-2 * eye(12);

    elseif robotName == "toy25"
        % Original demo setting.
        wbcParams.Wforce_per_foot = diag([1.0, 1.0, 3.0]);
        wbcParams.Wtau  = 1e-3 * eye(12);
        wbcParams.Wdtau = 1e-2 * eye(12);
    end

    %% Impedance dependent parameters
    % Normalize force mismatch by weight per leg.
    impParams.F0 = max(mass * g / 4, 10.0);

    % Scale lookup-table impedance gains.
    impParams = scaleImpedanceTables(impParams, impScale);

    % Scale clip limits too.
    impParams.Kp_min = impScale * impParams.Kp_min;
    impParams.Kp_max = impScale * impParams.Kp_max;
    impParams.Kd_min = impScale * impParams.Kd_min;
    impParams.Kd_max = impScale * impParams.Kd_max;

    % Conservative residual strength for smaller real-ish platforms.
    if robotName == "go1"
        impParams.lambda_res = min(impParams.lambda_res, 0.15);
        impParams.alpha_slip = min(impParams.alpha_slip, 2.0);
        impParams.beta_slip  = min(impParams.beta_slip, 2.0);
    elseif robotName == "spot"
        impParams.lambda_res = min(impParams.lambda_res, 0.12);
        impParams.alpha_slip = min(impParams.alpha_slip, 3.0);
        impParams.beta_slip  = min(impParams.beta_slip, 3.0);
    end

    %% Metadata
    meta.mass = mass;
    meta.g = g;
    meta.mg = mass*g;
    meta.Ibody = Ibody;
    meta.hip_offset_body = hip_offset_body;
    meta.p_foot_now = p_foot_now;
    meta.nominal_body_height = nominal_body_height;
    meta.tau_max = tau_max;
    meta.F0 = impParams.F0;
    meta.impScale = impScale;
end

function impParams = scaleImpedanceTables(impParams, scale)
    modes = ["cons", "agg"];
    contacts = ["st", "sw"];

    for mi = 1:numel(modes)
        m = modes(mi);
        for ci = 1:numel(contacts)
            c = contacts(ci);
            impParams.Kp.(m).(c) = scale * impParams.Kp.(m).(c);
            impParams.Kd.(m).(c) = scale * impParams.Kd.(m).(c);
        end
    end
end
