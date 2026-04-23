function IMP = impedanceResidualV2(z_t, Theta, Ref, WBC, params)
% impedanceResidualV2.m
%
% Version-2 impedance residual:
%   Kp_eff = Kp_nom - phase_gate * beta_imp  * tanh(s_imp/s_imp0) * I
%   Kd_eff = Kd_nom + phase_gate * alpha_imp * tanh(s_imp/s_imp0) * I
%
% Difference from Version 1:
%   Version 1 only increased damping.
%   Version 2 also decreases stiffness near touchdown when impact is detected.
%
% Residual torque:
%   tau_res = sum_i gamma_i * Jfi_j' * (Kp_eff*e_x + Kd_eff*e_v)
%
% Final torque:
%   tau_final = tau_OC + lambda_res * tau_res

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

    tau_res = zeros(12,1);

    impact_signal = zeros(4,1);
    phase_gate = zeros(4,1);
    gamma_leg = zeros(4,1);
    Kp_delta_diag = zeros(4,1);
    Kd_delta_diag = zeros(4,1);

    legDiag = struct();

    for leg = 1:4
        Si = S0(leg);
        phi_i = phase0(leg);

        [Kp_base, Kd_base] = lookupImpedanceBase(z_t, Si, params);

        Kp_nom = Theta.ctrl.k_des * Kp_base;
        Kd_nom = Theta.ctrl.k_des * Kd_base;

        % Positive impact signal when foot moves downward.
        vz_foot = xd_foot_now(3,leg);
        s_imp = max(0.0, -vz_foot);

        gate = touchdownGate(phi_i, params.touchdown_phase, params.phase_width);
        sat_imp = tanh(s_imp / params.s_imp0);

        % Version 2 adaptation:
        %   decrease stiffness, increase damping near touchdown impact.
        dKp_scalar = -gate * params.beta_imp  * sat_imp;
        dKd_scalar =  gate * params.alpha_imp * sat_imp;

        dKp = dKp_scalar * eye(3);
        dKd = dKd_scalar * eye(3);

        Kp_eff = clipDiagMatrix(Kp_nom + dKp, params.Kp_min, params.Kp_max);
        Kd_eff = clipDiagMatrix(Kd_nom + dKd, params.Kd_min, params.Kd_max);

        e_x = x_foot_ref(:,leg)  - x_foot_now(:,leg);
        e_v = xd_foot_ref(:,leg) - xd_foot_now(:,leg);

        F_imp = Kp_eff * e_x + Kd_eff * e_v;

        Jj = mockFootJacobianJointBlock(leg);
        tau_i = Jj' * F_imp;

        % Small baseline plus touchdown emphasis.
        gamma_i = 0.20 + 0.80 * gate;

        idx = (leg-1)*3 + (1:3);
        tau_res(idx) = tau_res(idx) + gamma_i * tau_i;

        impact_signal(leg) = s_imp;
        phase_gate(leg) = gate;
        gamma_leg(leg) = gamma_i;
        Kp_delta_diag(leg) = mean(diag(dKp));
        Kd_delta_diag(leg) = mean(diag(dKd));

        legDiag(leg).S = Si;
        legDiag(leg).phase = phi_i;
        legDiag(leg).impact = s_imp;
        legDiag(leg).gate = gate;
        legDiag(leg).gamma = gamma_i;
        legDiag(leg).dKp_scalar = dKp_scalar;
        legDiag(leg).dKd_scalar = dKd_scalar;
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
    IMP.gamma_leg = gamma_leg;
    IMP.Kp_delta_diag = Kp_delta_diag;
    IMP.Kd_delta_diag = Kd_delta_diag;
    IMP.leg = legDiag;
end

function [Kp, Kd] = lookupImpedanceBase(z_t, S_i, params)
    if z_t == 0
        if S_i == 1
            Kp = params.Kp.cons.st; Kd = params.Kd.cons.st;
        else
            Kp = params.Kp.cons.sw; Kd = params.Kd.cons.sw;
        end
    elseif z_t == 1
        if S_i == 1
            Kp = params.Kp.agg.st; Kd = params.Kd.agg.st;
        else
            Kp = params.Kp.agg.sw; Kd = params.Kd.agg.sw;
        end
    else
        error("Invalid z_t. Use 0 for conservative, 1 for aggressive.");
    end
end

function gate = touchdownGate(phi, phi_td, width)
% Gaussian-shaped phase gate near touchdown.
% Uses circular distance because phase wraps in [0,1).
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
