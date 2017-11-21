clear all; 
close all; 
clc;

alpha = 1/3;
beta = 0.99;
sigma = 2;
delta = 0.025;
a_lo = 0;
rho = 0.5;
sigma_e = 0.2;

% discretize z
m = 5;
[ln_z, PI] = TAUCHEN(m,rho,sigma_e,3); 
PI_inv = PI^100;
prob = PI_inv(1,:);
z = exp(ln_z); 
N_s = prob*z; 

n = 500;
a_min = a_lo;
a_max = 100; 
a = linspace(a_min, a_max, n);

K_min = 0;
K_max = 100;
dis1 = 1;
tol1 = 0.01;

while abs(dis1)>= tol1
    K_guess = (K_max + K_min)/2;
    r = alpha*K_guess^(alpha-1)*N_s^(1-alpha)+(1-delta);
    w = (1-alpha)*(K_guess^alpha)*N_s^(-alpha);
    % current return function
    cons = bsxfun(@plus,bsxfun(@minus,r*a',a),permute(w*z', [1 3 2]));
    ret = (cons .^ (1-sigma)) ./ (1 - sigma);
    ret (cons < 0) = -Inf;

    v_guess = zeros(m,n);

    dis2 = 1;
    tol2 = 1e-06;
    while dis2 > tol2
        % CONSTRUCT RETURN + EXPECTED CONTINUATION VALUE
        v = ret + beta * repmat(permute((PI*v_guess),[3 2 1]),[n 1 1]);        
        % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
        [vfn, p_indx] = max(v, [], 2);
        dis2 = max(max(abs(permute(vfn, [3 1 2])-v_guess)));        
        v_guess = permute(vfn, [3 1 2]);
    end
    pol_indx = permute(p_indx,[3,1,2]);
    pol_fn = a(pol_indx);
    Mu = ones(m,n)/(m*n);
    dis3 = 1;
    tol3 = 1e-06;
    while dis3 >= tol3
        [emp_ind, a_ind, mass] = find(Mu); 
        MuNew = zeros(size(Mu));
        for ii = 1:length(emp_ind)
            % which a prime does the policy fn prescribe?
            apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); 
            % which mass of households goes to which exogenous state?
            MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
                (PI(emp_ind(ii), :) * mass(ii))';
        end
        dis3 = max(max(abs(MuNew-Mu)));
        Mu = MuNew;
    end
    %check for market clearing
    aggsav = sum(sum(Mu.*pol_fn));
    dis1 = aggsav - K_guess;

    if dis1 > 0
        K_min = K_guess;
    else
        K_max = K_guess;
    end
end
K = K_guess;
display(['The steady state capital stock is ',num2str(K)]);

% interest rates
display(['Steady state interest rate is ', num2str(r)]);
r_CM = 1/beta;
display(['The interest rate in complete market is ', num2str(r_CM)]);

% Policy Functions
figure
plot(a,pol_fn)
legend('z = 0.5002','z = 0.7072','z = 1','z = 1.4140', 'z = 1.9993')
title('The Policy Functions')

% Lorenz Curve and Gini Coeff.
d = reshape(Mu',[m*n 1]);
wealth = reshape(repmat(a, [m 1])',[m*n 1]);
d_wealth = cumsum(sortrows([d,d.*wealth,wealth],3));
G_wealth = bsxfun(@rdivide,d_wealth,d_wealth(end,:));
L_wealth = G_wealth*100;
Gini_wealth = 1-(sum((G_wealth(1:end-1,2)+G_wealth(2:end,2)).*...
    diff(G_wealth(:,1))));
display(['The Gini coefficient for wealth is ', num2str(Gini_wealth)]);
figure
plot(L_wealth(:,1),L_wealth(:,2),L_wealth(:,1),L_wealth(:,1),'--k')
legend('Lorenz Curve','45 degree line')
title('Lorenz Curve for Wealth')