function Theta = thetaDecoder(z_t, a_HL, x_hat, u_cmd)
% thetaDecoder.m
%
% Rule-based Theta decoder prototype.
%
% Inputs
%   z_t   : 0 = conservative, 1 = aggressive
%   a_HL  : [a_swing, a_body, a_duty, a_imp]'
%   x_hat : [x y z roll pitch yaw vx vy vz wx wy wz]'
%   u_cmd : [vx_cmd vy_cmd wz_cmd]'

    a_swing = a_HL(1);
    a_body  = a_HL(2);
    a_duty  = a_HL(3);
    a_imp   = a_HL(4);

    roll  = x_hat(4);
    pitch = x_hat(5);
    vx    = x_hat(7);
    vy    = x_hat(8);

    vx_cmd = u_cmd(1);
    vy_cmd = u_cmd(2);

    v_cmd_mag = sqrt(vx_cmd^2 + vy_cmd^2);
    v_err_xy  = [vx - vx_cmd; vy - vy_cmd; 0.0];

    clip = @(x, xmin, xmax) min(max(x, xmin), xmax);

    if z_t == 0
        phase_nom = [0.00; 0.25; 0.50; 0.75];
        T_nom     = 0.75;
        duty_nom  = 0.78;

        h_body_nom  = 0.36;
        h_swing_nom = 0.07;

        k_imp_nom = 0.80;
        mu_nom    = 0.55;

        Kv_foot = diag([-0.08, -0.08, 0.0]);

    elseif z_t == 1
        phase_nom = [0.00; 0.50; 0.50; 0.00];
        T_nom     = 0.45;
        duty_nom  = 0.55;

        h_body_nom  = 0.43;
        h_swing_nom = 0.11;

        k_imp_nom = 1.15;
        mu_nom    = 0.60;

        Kv_foot = diag([-0.12, -0.12, 0.0]);

    else
        error("Invalid z_t. Use 0 for conservative, 1 for aggressive.");
    end

    k_swing = 0.04;
    k_body  = 0.05;
    k_duty  = 0.10;
    k_imp   = 0.25;

    duty_state_corr = 0.05 * abs(roll) + 0.05 * abs(pitch);
    body_state_corr = -0.03 * abs(roll) - 0.03 * abs(pitch);

    phase_i = phase_nom;

    T_raw = T_nom * (1.0 - 0.25 * tanh(v_cmd_mag));
    T = clip(T_raw, 0.30, 1.00);

    duty = duty_nom + k_duty * a_duty + duty_state_corr;
    duty = clip(duty, 0.35, 0.90);
    duty_i = duty * ones(4,1);

    h_body = h_body_nom + k_body * a_body + body_state_corr;
    h_body = clip(h_body, 0.25, 0.55);

    roll_ref  = 0.0;
    pitch_ref = 0.0;

    h_swing = h_swing_nom + k_swing * a_swing + 0.02 * tanh(v_cmd_mag);
    h_swing = clip(h_swing, 0.03, 0.18);
    h_swing_i = h_swing * ones(4,1);

    delta_p_td = Kv_foot * v_err_xy;
    delta_p_td(1) = clip(delta_p_td(1), -0.08, 0.08);
    delta_p_td(2) = clip(delta_p_td(2), -0.05, 0.05);
    delta_p_td(3) = 0.0;
    delta_p_td_i = repmat(delta_p_td, 1, 4);

    delta_t_td_i = zeros(4,1);

    k_des = k_imp_nom + k_imp * a_imp;
    k_des = clip(k_des, 0.40, 1.60);

    mu_exp = clip(mu_nom, 0.30, 0.90);

    Theta.gait.phase_i = phase_i;
    Theta.gait.T       = T;
    Theta.gait.duty_i  = duty_i;

    Theta.foot.delta_p_td_i = delta_p_td_i;
    Theta.foot.h_swing_i    = h_swing_i;
    Theta.foot.delta_t_td_i = delta_t_td_i;

    Theta.base.h_body_ref = h_body;
    Theta.base.roll_ref   = roll_ref;
    Theta.base.pitch_ref  = pitch_ref;

    Theta.ctrl.k_des  = k_des;
    Theta.ctrl.mu_exp = mu_exp;
end
