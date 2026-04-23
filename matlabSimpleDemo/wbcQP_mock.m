function WBC = wbcQP_mock(x_hat, qj, dqj, Ref, Theta, MPC, params)
% wbcQP_mock.m
%
% Mock WBC QP to verify interface and matrix assembly.
%
% Decision variable:
%   y = [qddot; tau; f_c]
%
%   qddot : 18 x 1 generalized acceleration
%   tau   : 12 x 1 joint torque
%   f_c   : 3*n_c x 1 stance contact forces
%
% This prototype uses simplified/mock dynamics and task Jacobians.
% Replace mockRobotModel() with real robot dynamics from URDF later.

    S0 = Ref.S(:,1); % 4 x 1, 1=stance, 0=swing
    stanceLegs = find(S0 == 1);
    swingLegs  = find(S0 == 0);
    nc = numel(stanceLegs);

    nq = 18; ntau = 12; nf = 3*nc; nY = nq + ntau + nf;

    model = mockRobotModel(x_hat, qj, dqj, Ref, stanceLegs, params);

    M = model.M;
    h = model.h;
    Sact = model.Sact;
    Jc = model.Jc;
    Jcdqdot = model.Jcdqdot;
    Jb = model.Jb;
    Jbdqdot = model.Jbdqdot;
    Jfoot = model.Jfoot;
    Jfootdqdot = model.Jfootdqdot;
    p_foot_now = model.p_foot_now;
    v_foot_now = model.v_foot_now;

    %% Base task
    xb_ref = Ref.Xb_ref(:,1);
    xb_actual = x_hat(1:6);
    dxb_actual = x_hat(7:12);
    xb_pos_ref = xb_ref(1:6);
    xb_vel_ref = xb_ref(7:12);
    xbdd_ref = zeros(6,1);

    xbdd_des = xbdd_ref + params.Kp_base * (xb_pos_ref - xb_actual) ...
                        + params.Kd_base * (xb_vel_ref - dxb_actual);

    Ab = [Jb, zeros(6,ntau), zeros(6,nf)];
    bb = xbdd_des - Jbdqdot;

    %% Swing foot tasks
    Asw = []; bsw = []; Wsw_blocks = {};
    for idx = 1:numel(swingLegs)
        leg = swingLegs(idx);
        Jfi = Jfoot{leg};
        Jfidqdot = Jfootdqdot{leg};

        xf_ref = Ref.Xf_ref(:,leg,1);
        vf_ref = Ref.Xfd_ref(:,leg,1);
        af_ref = zeros(3,1);

        xf_now = p_foot_now(:,leg);
        vf_now = v_foot_now(:,leg);

        af_des = af_ref + params.Kp_foot * (xf_ref - xf_now) ...
                         + params.Kd_foot * (vf_ref - vf_now);

        Afi = [Jfi, zeros(3,ntau), zeros(3,nf)];
        bfi = af_des - Jfidqdot;

        Asw = [Asw; Afi]; %#ok<AGROW>
        bsw = [bsw; bfi]; %#ok<AGROW>
        Wsw_blocks{end+1} = params.Wfoot; %#ok<AGROW>
    end
    if isempty(Asw)
        Asw = zeros(0,nY); bsw = zeros(0,1); Wsw = zeros(0,0);
    else
        Wsw = blkdiag(Wsw_blocks{:});
    end

    %% Force tracking task: stance force only
    f_full_star = MPC.f_t_star;
    f_c_star = zeros(nf,1);
    for j = 1:nc
        leg = stanceLegs(j);
        f_c_star((j-1)*3 + (1:3)) = f_full_star((leg-1)*3 + (1:3));
    end
    AF = [zeros(nf,nq), zeros(nf,ntau), eye(nf)];
    bF = f_c_star;

    WF_blocks = cell(1,nc);
    for j = 1:nc
        WF_blocks{j} = params.Wforce_per_foot;
    end
    if nc > 0
        WF = blkdiag(WF_blocks{:});
    else
        WF = zeros(0,0);
    end

    %% Torque magnitude and torque-rate tasks
    Atau = [zeros(ntau,nq), eye(ntau), zeros(ntau,nf)];
    btau = zeros(ntau,1);

    Adtau = Atau;
    bdtau = params.tau_prev;

    %% Stack weighted least squares objective
    Atask = [Ab; Asw; AF; Atau; Adtau];
    btask = [bb; bsw; bF; btau; bdtau];
    Wtask = blkdiag(params.Wb, Wsw, WF, params.Wtau, params.Wdtau);

    H = 2 * (Atask' * Wtask * Atask);
    g = -2 * (Atask' * Wtask * btask);
    H = 0.5 * (H + H') + params.H_reg * eye(size(H));

    %% Equality constraints
    % Full-body dynamics: M*qddot - S^T*tau - Jc^T*f = -h
    A_dyn = [M, -Sact', -Jc'];
    b_dyn = -h;

    % Stance foot no acceleration: Jc*qddot = -Jcdot*qdot
    A_contact = [Jc, zeros(3*nc,ntau), zeros(3*nc,nf)];
    b_contact = -Jcdqdot;

    Aeq = [A_dyn; A_contact];
    beq = [b_dyn; b_contact];

    %% Inequality constraints
    mu = params.mu;
    if isempty(mu)
        mu = Theta.ctrl.mu_exp;
    end
    [Cf, df] = buildFrictionForces(nc, mu, params.fz_max_per_leg);
    A_fric = [zeros(size(Cf,1), nq), zeros(size(Cf,1), ntau), Cf];
    b_fric = df;

    A_tau_lim = [
        zeros(ntau,nq),  eye(ntau), zeros(ntau,nf);
        zeros(ntau,nq), -eye(ntau), zeros(ntau,nf)
    ];
    b_tau_lim = [params.tau_max; -params.tau_min];

    Aineq = [A_fric; A_tau_lim];
    bineq = [b_fric; b_tau_lim];

    %% Solve QP
    options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
    try
        [y_star, fval, exitflag, output] = quadprog(H, g, Aineq, bineq, Aeq, beq, [], [], [], options);
    catch ME
        y_star = nan(nY,1); fval = nan; exitflag = -999;
        output.message = "quadprog failed: " + string(ME.message);
    end

    if any(isnan(y_star))
        qddot_star = nan(nq,1); tau_star = nan(ntau,1); f_c = nan(nf,1);
    else
        qddot_star = y_star(1:nq);
        tau_star = y_star(nq+1:nq+ntau);
        f_c = y_star(nq+ntau+1:end);
    end

    WBC.y_star = y_star;
    WBC.qddot = qddot_star;
    WBC.tau_OC = tau_star;
    WBC.f_c = f_c;
    WBC.stanceLegs = stanceLegs;
    WBC.swingLegs = swingLegs;
    WBC.H = H; WBC.g = g;
    WBC.Aeq = Aeq; WBC.beq = beq;
    WBC.Aineq = Aineq; WBC.bineq = bineq;
    WBC.fval = fval; WBC.exitflag = exitflag;
    if exist('output','var') && isfield(output,'message')
        WBC.message = output.message;
    else
        WBC.message = "";
    end
end

function model = mockRobotModel(x_hat, qj, dqj, Ref, stanceLegs, params)
% Simplified model for QP assembly testing only.
% Replace with URDF-derived dynamics later.

    m = params.robot.mass;
    g = params.robot.g;
    Ibody = params.robot.Ibody;
    nq = 18; nc = numel(stanceLegs);

    M = eye(nq);
    M(1:3,1:3) = m * eye(3);
    M(4:6,4:6) = Ibody;
    M(7:18,7:18) = 0.25 * eye(12);

    h = zeros(nq,1);
    h(3) = m * g;

    Sact = [zeros(12,6), eye(12)];

    Jc = zeros(3*nc, nq);
    Jcdqdot = zeros(3*nc,1);
    for j = 1:nc
        leg = stanceLegs(j);
        rows = (j-1)*3 + (1:3);
        Jleg = mockFootJacobianForLeg(leg);
        Jc(rows, 1:3) = eye(3);
        Jc(rows, 6 + (leg-1)*3 + (1:3)) = Jleg;
    end

    Jb = [eye(6), zeros(6,12)];
    Jbdqdot = zeros(6,1);

    Jfoot = cell(4,1);
    Jfootdqdot = cell(4,1);
    for leg = 1:4
        Jfi = zeros(3,nq);
        Jfi(:,1:3) = eye(3);
        Jfi(:,6 + (leg-1)*3 + (1:3)) = mockFootJacobianForLeg(leg);
        Jfoot{leg} = Jfi;
        Jfootdqdot{leg} = zeros(3,1);
    end

    model.M = M; model.h = h; model.Sact = Sact;
    model.Jc = Jc; model.Jcdqdot = Jcdqdot;
    model.Jb = Jb; model.Jbdqdot = Jbdqdot;
    model.Jfoot = Jfoot; model.Jfootdqdot = Jfootdqdot;
    model.p_foot_now = Ref.Xf_ref(:,:,1);
    model.v_foot_now = zeros(3,4);
    model.qdot_full = [x_hat(7:9); x_hat(10:12); dqj];
end

function Jleg = mockFootJacobianForLeg(leg)
    signY = 1;
    if leg == 2 || leg == 4
        signY = -1;
    end
    Jleg = [
        0.00,  0.18,  0.10;
        0.12*signY, 0.00, 0.00;
        0.00, -0.22, -0.20
    ];
end

function [Cf, df] = buildFrictionForces(nc, mu, fz_max)
    Cfoot = [
         1,  0, -mu;
        -1,  0, -mu;
         0,  1, -mu;
         0, -1, -mu;
         0,  0, -1;
         0,  0,  1
    ];
    dfoot = [0; 0; 0; 0; 0; fz_max];

    Cf = zeros(6*nc, 3*nc);
    df = zeros(6*nc, 1);
    for j = 1:nc
        rows = (j-1)*6 + (1:6);
        cols = (j-1)*3 + (1:3);
        Cf(rows, cols) = Cfoot;
        df(rows) = dfoot;
    end
end
