function [sass_optimizer, sass_config] = SASS()
%SASS A stochastic hillclimber for real spaces
%  This module is expected to be used within UEGO
    sass_optimizer = @SASS_optimizer;
    sass_config = buildConfig();
end

% Internal auxiliary functions:
function config = buildConfig()
    config.name = 'SASS'; % Default configuration to work with
    config.Scnt = 5; % Minimum number of success
    config.Fcnt = 3; % Minimum number of failures
    config.ex = 2.0; % Expansion
    config.ct = 0.5; % Constraction
    config.sig_ub = 1.0; % Upper perturbation limit
    config.sig_lb = 0.00001; % Lower perturbation limit
    config.MaxFcnt = 32; % Maximum number of consecutive fails to stop (included from V. Plaza's TFM)
end

function [x, current_value] = SASS_optimizer(start_point, radius, initial_val, bounds, func, max_evals, optimizer_config)
    x = start_point;
    current_value = initial_val;
    dim = size(bounds, 1); % Number of dimensions
    b = zeros(dim, 1); % Bias
    k = 0; % Iteration counter
    scnt = 0; % Success counter
    fcnt = 0; % Failure counter
    sigma = optimizer_config.sig_ub;
    while k<max_evals && fcnt < optimizer_config.MaxFcnt
        if sigma < optimizer_config.sig_lb || sigma > optimizer_config.sig_ub % The sigma > ub is included in Jelasity's...
            sigma = optimizer_config.sig_ub; % scnt = 0;% Jelasity fcnt = 0;% Jelasity b = 0*b;% Jelasity (Restart)
        end
        if scnt > optimizer_config.Scnt
            sigma = optimizer_config.ex * sigma;
        end
        if fcnt > optimizer_config.Fcnt
            sigma = optimizer_config.ct * sigma;
        end
        % Generate a multivariate Gaussian vector (perturbation):
        xi = b + normrnd(0, sigma, [dim, 1]);
        factor = abs(normrnd(0, sigma/(optimizer_config.sig_ub*3))); % 99.7% of a population is within +/- 3 standard deviations
        if factor > 1.0
           factor = 1.0;
        end
        xi = factor*radius*( xi / norm(xi) ); % Re-scaling
        x_prime = x + xi;
        x_prime(x_prime < bounds(:, 1)) = bounds(x_prime < bounds(:, 1), 1); % If too small: saturate
        x_prime(x_prime > bounds(:, 2)) = bounds(x_prime > bounds(:, 2), 2); % If too big: saturate
        val_prime = func(x_prime);
        k = k + 1; % Add a new function evaluation 
        if val_prime < current_value % Move:
            x = x_prime;
            current_value = val_prime;
            b = 0.2*b + 0.4*xi;
            scnt = scnt+1;
            fcnt = 0;
        elseif k<max_evals % As long as we can call func one more time, continue
            x_prime = x - xi;
            x_prime(x_prime < bounds(:, 1)) = bounds(x_prime < bounds(:, 1), 1); % If too small: saturate
            x_prime(x_prime > bounds(:, 2)) = bounds(x_prime > bounds(:, 2), 2); % If too big: saturate
            val_prime = func(x_prime);
            k = k + 1; % Add a new function evaluation 
            if val_prime < current_value
                x = x_prime;
                current_value = val_prime;
                b = b - 0.4*xi;
                scnt = scnt+1; 
                fcnt = 0;
            else
                b = 0.5*b;
                fcnt = fcnt+1; 
                scnt = 0;
            end
        end
    end
end
