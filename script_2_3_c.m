% Initial w (6x1 vector of zeros)
w0 = zeros(6, 1);

% Tolerance
epsi = 1e-6;

% Run BFGS
[w_star, f_star, k] = bfgs('obj_fun', 'grad_fun', w0, epsi);


% Test sample 1
xt1 = [-0.25; -0.26];
zt1 = [xt1(1); xt1(2); xt1(1)^2; xt1(1)*xt1(2); xt1(2)^2; 1];
y_pred1 = sign(w_star' * zt1)

% Test sample 2
xt2 = [2; -0.3];
zt2 = [xt2(1); xt2(2); xt2(1)^2; xt2(1)*xt2(2); xt2(2)^2; 1];
y_pred2 = sign(w_star' * zt2)