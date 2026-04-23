function IMP = impedanceResidualV3(z_t, Theta, Ref, WBC, MPC, params)
% impedanceResidualV3.m
%
% Version-3 impedance residual:
%   Adds impact, force-mismatch, slip, and mid-swing tracking adaptation.
%
% Main ideas:
%   1) Near touchdown impact:
%        Kp down, Kd up
%   2) Contact-force mismatch:
%        Kp down, Kd up
%   3) Stance slip:
%        tangential Kp down, Kd up
%   4) Mid-swing tracking error:
%        Kp up, but only away from touchdown
%
% Final torque:
%   tau_final = tau_OC + lambda_res * tau_res
%
% This demo still uses mock foot states/Jacobians.
% Replace x_foot_now, xd_foot_now, and Jj with FK/Jacobian from the real robot later.

    tau_OC = WBC.tau_OC;

    S0 = Ref.S(:,1);       % 4 x 1, 1=stance, 0=swing
    phase0 = Ref.phase(:,1);

    x_foot_ref  = Ref.Xf_ref(:,:,1);
    xd_foot_ref = Ref.Xfd_ref(:,:,1);

    % Mock actual foot state.
    % x_now = x_ref - error, so e_x = x_ref - x_now = error.
    e_pos_mock = params.mock_position_error;
    x_foot_now = x_foot_ref - e_pos_mock;

    xd_foot_now = params.mock_foot_velocity_now;

    % Build estimated/contact force proxy from WBC optimized stance force.
    % In a real robot, f_est should come from contact estimator / force sensor / torque-based estimation.
    f_est_full = zeros(12,1);
    if isfield(WBC, "stanceLegs") && isfield(WBC, "f_c") && ~any(isnan(WBC.f_c))
        for j = 1:numel(WBC.stanceLegs)
            leg = WBC.stanceLegs(j);
            f_est_full((leg-1)*3 + (1:3)) = WBC.f_c((j-1)*3 + (1:3));
        end
    end
    f_mpc_full = MPC.f_t_star;

    tau_res = zeros(12,1);

    impact_signal = zeros(4,1);
    phase_gate = zeros(4,1);
    slip_signal = zeros(4,1);
    force_mismatch = zeros(4,1);
    force_mismatch_raw = zeros(4,1);
    track_error = zeros(4,1);
    gamma_leg = zeros(4,1);

    Kp_delta_diag = zeros(4,1);
    Kd_delta_diag = zeros(4,1);

    legDiag = struct();

    for leg = 1:4
        Si = S0(leg);
        phi_i = phase0(leg);

        % Nominal impedance from lookup table.
        [Kp_base, Kd_base] = lookupImpedanceBase(z_t, Si, params);

        % k_des scales nominal impedance.
        Kp_nom = Theta.ctrl.k_des * Kp_base;
        Kd_nom = Theta.ctrl.k_des * Kd_base;

        % --- Signals -----------------------------------------------------
        % Impact signal: positive when foot moves downward.
        vz_foot = xd_foot_now(3,leg);
        s_imp = max(0.0, -vz_foot);

        % Phase gate near touchdown.
        gate_td = touchdownGate(phi_i, params.touchdown_phase, params.phase_width);

        % Stance/swing indicators.
        is_stance = double(Si == 1);
        is_swing  = double(Si == 0);

        % Slip signal: tangential foot speed, only meaningful in stance.
        vxy_foot = xd_foot_now(1:2,leg);
        s_slip = norm(vxy_foot) * is_stance;

%         % Force mismatch signal.
%         f_mpc_i = f_mpc_full((leg-1)*3 + (1:3));
%         f_est_i = f_est_full((leg-1)*3 + (1:3));
%         eF_i = f_mpc_i - f_est_i;
%         s_force = norm(eF_i) * is_stance;
        
        % Force mismatch signal.
        f_mpc_i = f_mpc_full((leg-1)*3 + (1:3));
        f_est_i = f_est_full((leg-1)*3 + (1:3));
        eF_i = f_mpc_i - f_est_i;

        % Raw force mismatch [N]
        s_force_raw = norm(eF_i) * is_stance;

        % Normalized force mismatch [-]
        s_force_norm = s_force_raw / max(params.F0, 1e-6);

        % Foot tracking error.
        e_x = x_foot_ref(:,leg) - x_foot_now(:,leg);
        e_v = xd_foot_ref(:,leg) - xd_foot_now(:,leg);
        s_track = norm(e_x);

        % Saturated scalars.
        sat_imp   = tanh(s_imp   / params.s_imp0);
        %sat_force = tanh(s_force / params.F0);
        sat_force = tanh(s_force_norm);
        sat_slip  = tanh(s_slip  / params.slip0);
        sat_track = tanh(s_track / params.e0);

        % --- Gain adaptation --------------------------------------------
        % 1) Impact: near touchdown -> lower stiffness, increase damping.
        dKp_impact = -gate_td * params.beta_imp  * sat_imp * eye(3);
        dKd_impact =  gate_td * params.alpha_imp * sat_imp * eye(3);

        % 2) Force mismatch: only in stance -> lower stiffness, increase damping.
        dKp_force = -is_stance * params.beta_F  * sat_force * eye(3);
        dKd_force =  is_stance * params.alpha_F * sat_force * eye(3);

        % 3) Slip: only in stance -> lower tangential stiffness, increase damping.
        dKp_slip = -is_stance * params.beta_slip * sat_slip * diag([1,1,0]);
        dKd_slip =  is_stance * params.alpha_slip * sat_slip * eye(3);

        % 4) Tracking: only in swing and away from touchdown -> increase stiffness.
        % This avoids increasing stiffness right at touchdown.
        dKp_track = is_swing * (1.0 - gate_td) * params.alpha_track * sat_track * eye(3);
        dKd_track = zeros(3,3);

        dKp = dKp_impact + dKp_force + dKp_slip + dKp_track;
        dKd = dKd_impact + dKd_force + dKd_slip + dKd_track;

        Kp_eff = Kp_nom + dKp;
        Kd_eff = Kd_nom + dKd;

        Kp_eff = clipDiagMatrix(Kp_eff, params.Kp_min, params.Kp_max);
        Kd_eff = clipDiagMatrix(Kd_eff, params.Kd_min, params.Kd_max);

        % Task-space impedance force.
        F_imp = Kp_eff * e_x + Kd_eff * e_v;

        % Mock actuated joint Jacobian for foot leg.
        Jj = mockFootJacobianJointBlock(leg);
        tau_i = Jj' * F_imp;

        % Gate residual strength.
        % Use small baseline, touchdown emphasis, plus mild stance correction if slip/force mismatch exists.
        gamma_i = 0.15 + 0.65 * gate_td + 0.20 * is_stance * max(sat_force, sat_slip);
        gamma_i = min(max(gamma_i, 0.0), 1.0);

        idx = (leg-1)*3 + (1:3);
        tau_res(idx) = tau_res(idx) + gamma_i * tau_i;

        % Diagnostics.
        impact_signal(leg) = s_imp;
        phase_gate(leg) = gate_td;
        slip_signal(leg) = s_slip;
        % force_mismatch(leg) = s_force;
        force_mismatch_raw(leg) = s_force_raw;
        force_mismatch(leg) = s_force_norm;
        track_error(leg) = s_track;
        gamma_leg(leg) = gamma_i;

        Kp_delta_diag(leg) = mean(diag(dKp));
        Kd_delta_diag(leg) = mean(diag(dKd));

        legDiag(leg).S = Si;
        legDiag(leg).phase = phi_i;
        legDiag(leg).impact = s_imp;
        legDiag(leg).gate = gate_td;
        legDiag(leg).slip = s_slip;
        % legDiag(leg).force_mismatch = s_force;
        legDiag(leg).force_mismatch = s_force_norm;
        legDiag(leg).force_mismatch_raw = s_force_raw;
        legDiag(leg).force_mismatch_norm = s_force_norm;
        legDiag(leg).track_error = s_track;
        legDiag(leg).gamma = gamma_i;
        legDiag(leg).dKp_impact = dKp_impact;
        legDiag(leg).dKd_impact = dKd_impact;
        legDiag(leg).dKp_force = dKp_force;
        legDiag(leg).dKd_force = dKd_force;
        legDiag(leg).dKp_slip = dKp_slip;
        legDiag(leg).dKd_slip = dKd_slip;
        legDiag(leg).dKp_track = dKp_track;
        legDiag(leg).Kp_nom = Kp_nom;
        legDiag(leg).Kd_nom = Kd_nom;
        legDiag(leg).Kp_eff = Kp_eff;
        legDiag(leg).Kd_eff = Kd_eff;
        legDiag(leg).F_imp = F_imp;
        legDiag(leg).tau_i = tau_i;
    end

    tau_final = tau_OC + params.lambda_res * tau_res;

    IMP.tau_OC = tau_OC;
    IMP.tau_res = tau_res;
    IMP.tau_final = tau_final;

    IMP.impact_signal = impact_signal;
    IMP.phase_gate = phase_gate;
    IMP.slip_signal = slip_signal;
    IMP.force_mismatch = force_mismatch;
    IMP.force_mismatch_raw = force_mismatch_raw;
    IMP.force_mismatch_norm = force_mismatch;
    IMP.track_error = track_error;
    IMP.gamma_leg = gamma_leg;

    IMP.Kp_delta_diag = Kp_delta_diag;
    IMP.Kd_delta_diag = Kd_delta_diag;

    IMP.leg = legDiag;
end

%% ============================================================
% Helper functions
% ============================================================

function [Kp, Kd] = lookupImpedanceBase(z_t, S_i, params)
    if z_t == 0
        if S_i == 1
            Kp = params.Kp.cons.st;
            Kd = params.Kd.cons.st;
        else
            Kp = params.Kp.cons.sw;
            Kd = params.Kd.cons.sw;
        end
    elseif z_t == 1
        if S_i == 1
            Kp = params.Kp.agg.st;
            Kd = params.Kd.agg.st;
        else
            Kp = params.Kp.agg.sw;
            Kd = params.Kd.agg.sw;
        end
    else
        error("Invalid z_t. Use 0 for conservative, 1 for aggressive.");
    end
end

function gate = touchdownGate(phi, phi_td, width)
    d = abs(phi - phi_td);
    d = min(d, 1.0 - d);
    gate = exp(-(d^2) / (2 * width^2));
end

function Mclip = clipDiagMatrix(M, Mmin, Mmax)
    d = diag(M);
    dmin = diag(Mmin);
    dmax = diag(Mmax);
    dclip = min(max(d, dmin), dmax);
    Mclip = diag(dclip);
end

function Jj = mockFootJacobianJointBlock(leg)
    signY = 1;
    if leg == 2 || leg == 4
        signY = -1;
    end

    Jj = [
        0.00,  0.18,  0.10;
        0.12*signY, 0.00, 0.00;
        0.00, -0.22, -0.20
    ];
end
