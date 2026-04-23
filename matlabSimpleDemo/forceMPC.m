function MPC = forceMPC(x_hat, Ref, Theta, beta_t, params)
% forceMPC.m
%
% Constrained Force MPC for GRF generation.
%
% State ordering:
%   x = [x y z roll pitch yaw vx vy vz wx wy wz]'
%
% Decision variable:
%   Xi = [f_0; f_1; ...; f_{N-1}]
%   f_k = [f_LF; f_RF; f_LH; f_RH]
%   f_i = [fx_i; fy_i; fz_i]
%
% Requires Optimization Toolbox for quadprog.

    N  = params.N;
    dt = params.dt;

    nx = 12;
    nf_per_step = 12;
    nXi = nf_per_step * N;

    m = params.robot.mass;
    grav = params.robot.g;
    Ibody = params.robot.Ibody;

    mu = Theta.ctrl.mu_exp;

    %% Build time-varying discrete dynamics over horizon
    A_list = cell(N,1);
    B_list = cell(N,1);
    d_list = cell(N,1);

    for k = 1:N
        x_ref_k = Ref.Xb_ref(:,k);

        % Use yaw from reference trajectory for linearization.
        yaw_k = x_ref_k(6);

        % Use foot reference positions to approximate current contact points.
        p_base_k = x_ref_k(1:3);
        p_foot_k = Ref.Xf_ref(:,:,k);

        [Ac, Bc, dc] = centroidalAffineDynamics(yaw_k, p_base_k, p_foot_k, m, Ibody, grav);

        A_list{k} = eye(nx) + dt * Ac;
        B_list{k} = dt * Bc;
        d_list{k} = dt * dc;
    end

    %% Build stacked prediction matrices
    [Abar, Bbar, dbar] = buildPredictionMatrices(A_list, B_list, d_list);

    Xref = reshape(Ref.Xb_ref, nx*N, 1);

    %% Build Q_t
    [wh, wv, we] = betaToWeights(beta_t, params.weightRange);
    Qbar = buildQbar(N, params.q0, wh, wv);

    %% Build force regularization matrices
    W_F = (1/(m*grav)^2) * eye(nXi);

    D = buildDifferenceMatrix(N, nf_per_step);
    W_DF = (1/(m*grav)^2) * eye(nf_per_step*(N-1));

    %% QP objective
    e0 = Abar * x_hat + dbar - Xref;

    H = 2 * (Bbar' * Qbar * Bbar + ...
        we * (params.rho_f * W_F + params.rho_df * (D' * W_DF * D)));

    g = 2 * Bbar' * Qbar * e0;

    H = 0.5 * (H + H') + params.H_reg * eye(size(H));

    %% Constraints
    [Aineq, bineq] = buildForceInequalityConstraints(N, mu, params.fz_max_per_leg);

    if isfield(params, "use_force_rate_bound") && params.use_force_rate_bound
        A_df = [D; -D];
        b_df = params.df_max * ones(size(A_df,1),1);
        Aineq = [Aineq; A_df];
        bineq = [bineq; b_df];
    end

    [Aeq, beq] = buildSwingEqualityConstraints(Ref.S, N);

    %% Solve QP
    options = optimoptions("quadprog", ...
        "Display", "off", ...
        "Algorithm", "interior-point-convex");

    try
        [Xi_star, fval, exitflag, output] = quadprog(H, g, Aineq, bineq, Aeq, beq, [], [], [], options);
    catch ME
        Xi_star = nan(nXi,1);
        fval = nan;
        exitflag = -999;
        output.message = "quadprog failed: " + string(ME.message);
    end

    %% Recover outputs
    if any(isnan(Xi_star))
        f_t_star = nan(12,1);
        X_pred = nan(nx,N);
        F_by_leg_z = nan(4,N);
    else
        f_t_star = Xi_star(1:12);
        X_stack = Abar * x_hat + Bbar * Xi_star + dbar;
        X_pred = reshape(X_stack, nx, N);

        F_by_leg_z = zeros(4,N);
        for k = 1:N
            fk = Xi_star((k-1)*12+1:k*12);
            Fmat = reshape(fk, 3, 4);
            F_by_leg_z(:,k) = Fmat(3,:)';
        end
    end

    %% Pack
    MPC.Xi_star = Xi_star;
    MPC.f_t_star = f_t_star;
    MPC.X_pred = X_pred;
    MPC.F_by_leg_z = F_by_leg_z;

    MPC.H = H;
    MPC.g = g;
    MPC.Aineq = Aineq;
    MPC.bineq = bineq;
    MPC.Aeq = Aeq;
    MPC.beq = beq;
    MPC.fval = fval;
    MPC.exitflag = exitflag;

    if exist("output", "var") && isfield(output, "message")
        MPC.message = output.message;
    else
        MPC.message = "";
    end

    MPC.weights.wh = wh;
    MPC.weights.wv = wv;
    MPC.weights.we = we;
end

%% ============================================================
% Local helper functions
% ============================================================

function [Ac, Bc, dc] = centroidalAffineDynamics(yaw, p_base, p_foot, m, Ibody, grav)
% Continuous-time affine dynamics:
%   x = [p; eta; v; omega]
%   f = [f1; f2; f3; f4]
%
%   p_dot     = v
%   eta_dot   = Rz(yaw)' * omega
%   v_dot     = (1/m) * sum_i f_i + g
%   omega_dot = I^{-1} * sum_i [r_i]_x f_i

    nx = 12;
    Ac = zeros(nx,nx);
    Bc = zeros(nx,12);
    dc = zeros(nx,1);

    Rz = rotz3d(yaw);

    % p_dot = v
    Ac(1:3, 7:9) = eye(3);

    % eta_dot = Rz' * omega
    Ac(4:6, 10:12) = Rz';

    % v_dot = sum f_i / m + g
    for i = 1:4
        col = (i-1)*3 + (1:3);
        Bc(7:9, col) = (1/m) * eye(3);
    end

    dc(7:9) = [0; 0; -grav];

    % omega_dot = I^{-1} sum [r_i]_x f_i
    Iinv = inv(Ibody);
    for i = 1:4
        col = (i-1)*3 + (1:3);
        r_i = p_foot(:,i) - p_base;
        Bc(10:12, col) = Iinv * skew(r_i);
    end
end

function [Abar, Bbar, dbar] = buildPredictionMatrices(A_list, B_list, d_list)
% Build stacked prediction:
%   X = Abar*x0 + Bbar*Xi + dbar

    N = numel(A_list);
    nx = size(A_list{1},1);
    nu = size(B_list{1},2);

    Abar = zeros(nx*N, nx);
    Bbar = zeros(nx*N, nu*N);
    dbar = zeros(nx*N, 1);

    for row = 1:N
        Aprod = eye(nx);
        for j = row:-1:1
            Aprod = Aprod * A_list{j};
        end
        Abar((row-1)*nx+1:row*nx, :) = Aprod;

        for col = 1:row
            Aprod_B = eye(nx);
            for j = row:-1:(col+1)
                Aprod_B = Aprod_B * A_list{j};
            end
            Bbar((row-1)*nx+1:row*nx, (col-1)*nu+1:col*nu) = Aprod_B * B_list{col};
        end

        d_accum = zeros(nx,1);
        for ell = 1:row
            Aprod_d = eye(nx);
            for j = row:-1:(ell+1)
                Aprod_d = Aprod_d * A_list{j};
            end
            d_accum = d_accum + Aprod_d * d_list{ell};
        end
        dbar((row-1)*nx+1:row*nx) = d_accum;
    end
end

function Qbar = buildQbar(N, q0, wh, wv)
% State ordering:
% [x y z roll pitch yaw vx vy vz wx wy wz]

    nx = 12;
    Qbar = zeros(nx*N, nx*N);

    q = zeros(12,1);

    q(1)  = q0.x;
    q(2)  = q0.y;
    q(3)  = wh * q0.z;

    q(4)  = q0.roll;
    q(5)  = q0.pitch;
    q(6)  = q0.yaw;

    q(7)  = wv * q0.vx;
    q(8)  = wv * q0.vy;
    q(9)  = q0.vz;

    q(10) = q0.wx;
    q(11) = q0.wy;
    q(12) = wv * q0.wz;

    Qk = diag(q);

    for k = 1:N
        idx = (k-1)*nx + (1:nx);
        Qbar(idx,idx) = Qk;
    end
end

function [wh, wv, we] = betaToWeights(beta, ranges)
% beta = [beta_h beta_v beta_e]'
% softmax -> bounded weights.

    b = beta(:);
    b = b - max(b);
    sm = exp(b) / sum(exp(b));

    bh = sm(1);
    bv = sm(2);
    be = sm(3);

    wh = ranges.wh(1) + (ranges.wh(2) - ranges.wh(1)) * bh;
    wv = ranges.wv(1) + (ranges.wv(2) - ranges.wv(1)) * bv;
    we = ranges.we(1) + (ranges.we(2) - ranges.we(1)) * be;
end

function D = buildDifferenceMatrix(N, n)
% D*Xi = [u_1-u_0; u_2-u_1; ...; u_{N-1}-u_{N-2}]
    D = zeros(n*(N-1), n*N);
    for k = 1:(N-1)
        row = (k-1)*n + (1:n);
        col_prev = (k-1)*n + (1:n);
        col_next = k*n + (1:n);
        D(row, col_prev) = -eye(n);
        D(row, col_next) = eye(n);
    end
end

function [Aineq, bineq] = buildForceInequalityConstraints(N, mu, fz_max)
% For each foot at each horizon:
%   fx <= mu fz
%  -fx <= mu fz
%   fy <= mu fz
%  -fy <= mu fz
%  -fz <= 0
%   fz <= fz_max

    n_rows_per_foot = 6;
    n_legs = 4;
    n_force_step = 12;

    Cfoot = [
         1,  0, -mu;
        -1,  0, -mu;
         0,  1, -mu;
         0, -1, -mu;
         0,  0, -1;
         0,  0,  1
    ];

    bfoot = [0; 0; 0; 0; 0; fz_max];

    rows_total = N * n_legs * n_rows_per_foot;
    cols_total = N * n_force_step;

    Aineq = zeros(rows_total, cols_total);
    bineq = zeros(rows_total, 1);

    row_ptr = 1;
    for k = 1:N
        for leg = 1:n_legs
            rows = row_ptr:(row_ptr+n_rows_per_foot-1);
            cols = (k-1)*n_force_step + (leg-1)*3 + (1:3);

            Aineq(rows, cols) = Cfoot;
            bineq(rows) = bfoot;

            row_ptr = row_ptr + n_rows_per_foot;
        end
    end
end

function [Aeq, beq] = buildSwingEqualityConstraints(S, N)
% S: 4 x N, 1=stance, 0=swing.
% If swing, enforce f_{i,k}=0.

    n_force_step = 12;
    swing_count = sum(S(:) == 0);

    Aeq = zeros(3*swing_count, n_force_step*N);
    beq = zeros(3*swing_count, 1);

    row_ptr = 1;
    for k = 1:N
        for leg = 1:4
            if S(leg,k) == 0
                rows = row_ptr:(row_ptr+2);
                cols = (k-1)*n_force_step + (leg-1)*3 + (1:3);

                Aeq(rows, cols) = eye(3);
                row_ptr = row_ptr + 3;
            end
        end
    end
end

function S = skew(r)
% Skew-symmetric matrix such that skew(r)*f = r x f.
    S = [
        0,    -r(3),  r(2);
        r(3),  0,    -r(1);
       -r(2),  r(1),  0
    ];
end

function R = rotz3d(yaw)
    c = cos(yaw);
    s = sin(yaw);
    R = [
        c, -s, 0;
        s,  c, 0;
        0,  0, 1
    ];
end
