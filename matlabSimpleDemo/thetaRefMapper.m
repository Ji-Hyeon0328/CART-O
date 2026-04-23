function Ref = thetaRefMapper(Theta, x_hat, u_cmd, params)
% thetaRefMapper.m
%
% Deterministic Theta-to-Reference Mapper.
%
% Outputs
%   Ref.S       : 4 x N contact schedule, 1=stance, 0=swing
%   Ref.phase   : 4 x N leg phase in [0,1)
%   Ref.Xb_ref  : 12 x N base reference state
%   Ref.Xf_ref  : 3 x 4 x N foot position reference
%   Ref.Xfd_ref : 3 x 4 x N foot velocity reference

    dt = params.dt;
    N  = params.N;

    p0   = x_hat(1:3);
    eta0 = x_hat(4:6);
    yaw0 = eta0(3);

    vx_cmd = u_cmd(1);
    vy_cmd = u_cmd(2);
    wz_cmd = u_cmd(3);

    phase_i = Theta.gait.phase_i(:);
    T       = Theta.gait.T;
    duty_i  = Theta.gait.duty_i(:);

    h_body_ref = Theta.base.h_body_ref;
    roll_ref   = Theta.base.roll_ref;
    pitch_ref  = Theta.base.pitch_ref;

    h_swing_i    = Theta.foot.h_swing_i(:);
    delta_p_td_i = Theta.foot.delta_p_td_i;
    delta_t_td_i = Theta.foot.delta_t_td_i(:);

    p_foot_now = params.p_foot_now;
    hip_offset_body = params.hip_offset_body;

    S       = zeros(4, N);
    phase   = zeros(4, N);
    Xb_ref  = zeros(12, N);
    Xf_ref  = zeros(3, 4, N);

    p_ref   = p0;
    yaw_ref = yaw0;

    for k = 1:N
        Xb_ref(:,k) = [
            p_ref(1);
            p_ref(2);
            h_body_ref;
            roll_ref;
            pitch_ref;
            yaw_ref;
            vx_cmd;
            vy_cmd;
            0.0;
            0.0;
            0.0;
            wz_cmd
        ];

        Rz = rotz2d(yaw_ref);
        v_world_xy = Rz * [vx_cmd; vy_cmd];

        p_ref(1) = p_ref(1) + dt * v_world_xy(1);
        p_ref(2) = p_ref(2) + dt * v_world_xy(2);
        p_ref(3) = h_body_ref;

        yaw_ref = yaw_ref + dt * wz_cmd;
    end

    for k = 1:N
        time_ahead = (k-1) * dt;

        p_base_k   = Xb_ref(1:3,k);
        yaw_base_k = Xb_ref(6,k);
        Rz_k       = rotz2d(yaw_base_k);

        for i = 1:4
            phi = mod(phase_i(i) + time_ahead / T, 1.0);
            phase(i,k) = phi;

            if phi < duty_i(i)
                S(i,k) = 1;
            else
                S(i,k) = 0;
            end

            hip_xy = p_base_k(1:2) + Rz_k * hip_offset_body(1:2,i);
            hip_z  = p_base_k(3) + hip_offset_body(3,i);
            p_hip_world = [hip_xy; hip_z];

            if S(i,k) == 1
                Xf_ref(:,i,k) = p_foot_now(:,i);
            else
                swing_denom = max(1e-6, 1.0 - duty_i(i));
                s = (phi - duty_i(i)) / swing_denom;
                s = min(max(s, 0.0), 1.0);

                T_st = duty_i(i) * T;
                p_td = p_hip_world + [
                    0.5 * T_st * vx_cmd;
                    0.5 * T_st * vy_cmd;
                    -h_body_ref
                ] + delta_p_td_i(:,i);

                if abs(delta_t_td_i(i)) > 1e-9
                    s = s / max(0.2, 1.0 + delta_t_td_i(i) / max(T, 1e-6));
                    s = min(max(s, 0.0), 1.0);
                end

                p_lo = p_foot_now(:,i);

                p_ref = (1-s) * p_lo + s * p_td;
                p_ref(3) = (1-s) * p_lo(3) + s * p_td(3) + h_swing_i(i) * sin(pi*s);

                Xf_ref(:,i,k) = p_ref;
            end
        end
    end

    Xfd_ref = zeros(3,4,N);
    if N >= 2
        for k = 2:N
            Xfd_ref(:,:,k) = (Xf_ref(:,:,k) - Xf_ref(:,:,k-1)) / dt;
        end
        Xfd_ref(:,:,1) = Xfd_ref(:,:,2);
    end

    Ref.S       = S;
    Ref.phase   = phase;
    Ref.Xb_ref  = Xb_ref;
    Ref.Xf_ref  = Xf_ref;
    Ref.Xfd_ref = Xfd_ref;
end

function R = rotz2d(yaw)
    c = cos(yaw);
    s = sin(yaw);
    R = [c, -s; s, c];
end
