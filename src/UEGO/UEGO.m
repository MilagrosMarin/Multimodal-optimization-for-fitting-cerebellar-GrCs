function [config, spec_list, spec_radii, spec_values] = UEGO(evals, levels, max_spec_num, min_r, bounds, func, local_optimizer, optimizer_config)
% UEGO  Implementation of Universal Evolutionary Global Optimizer (UEGO), 
% which is a stochastic population-based (memetic) optimization algorithm.
% [config, spec_list, spec_radii, spec_values] = UEGO(evals, levels, 
%   max_spec_num, min_r, bounds, func)
% Implementation by: N.C. Cruz (2019). Supervision by: J.L. Redondo, 
%   J.D. Alvarez, M. Berenguel and P.M. Ortigosa (UAL, Spain)
% See [M. Jelasity, P.M. Ortigosa & I. Garcia. UEGO, an abstract clustering 
% technique for multimodal global optimization. Journal of Heuristics, 7(3), 
% 215-233, 2001.] for further information about the method.
    config = ComputeConfig(evals, levels, max_spec_num, min_r, bounds);
    if ~isempty(config)
        [spec_list, spec_radii, spec_values] = Init_Spec_List(config.radii(1), bounds, func); % Level 1
        if config.n(1)>0
           [spec_list, spec_values] = Optimize_Species(spec_list, spec_radii, spec_values, bounds, func, config.n(1), local_optimizer, optimizer_config);
        end
        for i=2:1:levels
            [spec_list, spec_radii, spec_values] = Create_Species(spec_list, spec_radii, spec_values, bounds, func, config.new(i), config.radii(i));
            [spec_list, spec_radii, spec_values] = Fuse_Species(spec_list, spec_radii, spec_values, config.radii(i));
            [spec_list, spec_radii, spec_values] = Shorten_Spec_List(spec_list, spec_radii, spec_values, max_spec_num);
            [spec_list, spec_values] = Optimize_Species(spec_list, spec_radii, spec_values, bounds, func, fix(config.n(i)/max_spec_num), local_optimizer, optimizer_config);
            [spec_list, spec_radii, spec_values] = Fuse_Species(spec_list, spec_radii, spec_values, config.radii(i));
        end
        [spec_values, indices] = sort(spec_values, 'ascend'); % Sort the final list -> Lower is better
        spec_radii = spec_radii(indices);
        spec_list = spec_list(:, indices);
    end
end

% Internal auxiliary functions:
function val=speed(dim, r)
    V = [0.0, 0.25, 2/(3*pi), 3/16, 8/(15*pi), 10/64, 16/(35*pi), 35/256, 128/(315*pi), 63/512, 256/(693*pi), 231/2048, 1024/(3003*pi), 858/8192]; % Speed constants
    if dim < length(V)
        val = V(dim+1)*r; % WARNING: To access the same element as the C version
    else
       val = r / sqrt( pi * (2 * (dim+1)) );
    end
end

function config = ComputeConfig(evals, levels, max_spec_num, min_r, bounds)
    R_1 = norm(bounds(:,1) - bounds(:,2)); % Diameter of the search space
    config.radii = R_1 * ( (min_r / R_1) .^ ((0:1:(levels-1))/(levels-1))); % The radius of each level [1, L]
    config.radii(1) = R_1; % The radii of the first species is always the diameter
    
    config.new = 3*max_spec_num*ones(1, levels); % Func. evaluations for spec creation
    config.new(1) = 0; % (No new specs at the first level)
    
    evals1 = evals - (levels-1)*3*max_spec_num;
    dim = size(bounds, 1); % Number of dimensions
    if evals1>=0
        config.n = zeros(1, levels); % Func. evaluations for local optimization
        if levels<2
            config.n(1) = evals; % UEGO will not affect
        else
            config.n(1) = 0;
            accum = 0;
            for i=2:1:levels
                val = max_spec_num*R_1/speed(dim, config.radii(i));
                accum = accum + fix(val); % Warning: Discarding decimals as in the C version!!
                config.n(i) = val;
            end
            config.n(2:levels) = fix( config.n(2:levels)*(evals1/accum) ); % threshold (evals1/accum) % Warning: Discarding decimals as in the C version!!
        end
    else
        disp('Configuration error: All evaluations (N) too small, parameters not set.');
        config = []; 
    end
end

function [spec_list, spec_radii, spec_values] = Init_Spec_List(R_1, bounds, func)
    spec_list = (bounds(:,2)-bounds(:,1)) .* rand(size(bounds, 1), 1) + bounds(:,1); % Set of species (matrix). Each column is a solution point
    spec_radii = R_1; % The radius of the first species is equal to the diameter of the search space
    spec_values = func(spec_list(:, 1)); % Call the objective function
end

function new_point = RandomPoint(center, radius, bounds) % Function to randomly generate a new point from a species
    compass = (bounds(:,2) - bounds(:, 1)).*rand(size(bounds, 1), 1) + bounds(:, 1); % (max-min)*[0,1] + min; Random point in the search space
    dir = (compass - center);
    dist = norm(dir);
    if(dist<=radius) % The new point is already in range
        new_point = compass;
    else
        new_point = center + radius*rand()*(dir/dist); % Move from the center
        new_point(new_point < bounds(:, 1)) = bounds(new_point < bounds(:, 1), 1); % If too small: saturate
        new_point(new_point > bounds(:, 2)) = bounds(new_point > bounds(:, 2), 2); % If too big: saturate
    end
end

function [spec_list, spec_radii, spec_values] = Create_Species(spec_list, spec_radii, spec_values, bounds, func, evals, current_rad)
    num_species = size(spec_list, 2); % The number of columns is equal to the length of the list of species
    num_points = fix( (sqrt(1+8*(evals / num_species)) - 1)/2 ); % The number of points to generate per species
    
    pool = zeros(size(bounds, 1), num_points); % Where to save the new points of each species
    pool_values = zeros(1, num_points);
    insert_from_pool = zeros(1, num_points);
    
    for i=1:1:num_species % For each species:
        for j=1:1:num_points
            pool(:, j) = RandomPoint(spec_list(:, i), spec_radii(i), bounds); % New point
            pool_values(j) = func(pool(:, j));
        end
        insert_from_pool = 0*insert_from_pool; % Reset for this sublist
        for j=1:1:num_points % Making pairs
            if pool_values(j) < spec_values(i) % Replace the center of species 'i' (but not the radius)
                spec_list(:, i) = pool(:, j);
                spec_values(i) = pool_values(j);
            end
            for k=j+1:1:num_points
                middle = (pool(:, j) + pool(:, k))*0.5;
                middle_val = func(middle);
                if middle_val < spec_values(i) % Replace the center of species 'i' (but not the radius)
                    spec_list(:, i) = middle;
                    spec_values(i) = middle_val;
                end
                if middle_val > pool_values(j) && middle_val > pool_values(k) % The middle is worse than the values of the pair
                    insert_from_pool(j) = 1;
                    insert_from_pool(k) = 1; % Overriding sometimes...
                end
            end
        end
        spec_list = [spec_list pool(:, insert_from_pool>0)];
        spec_values = [spec_values pool_values(insert_from_pool>0)];
        spec_radii = [spec_radii repmat(current_rad, [1 sum(insert_from_pool)])];
    end
end

function [spec_list, spec_radii, spec_values] = Fuse_Species(spec_list, spec_radii, spec_values, current_rad)
    i = 1;
    i_restart = false;
    while i <= (size(spec_list, 2) - 1) % The number of columns is equal to the length of the list of species
        j = i + 1;
        while j <= size(spec_list, 2) && ~i_restart
            distance = norm(spec_list(:, i) - spec_list(:, j));
            if distance < current_rad % Fuse both species
                if spec_values(j) < spec_values(i) % That in j is a better center (there is no need to move otherwise)
                    spec_list(:, i) = spec_list(:, j);
                    spec_values(i) = spec_values(j);
                    i_restart = true;
                end
                if spec_radii(j) > spec_radii(i) % The radius of j is bigger: keep it
                    spec_radii(i) = spec_radii(j);
                end
                spec_list(:, j) = [];
                spec_values(j) = [];
                spec_radii(j) = [];
            else
               j = j + 1; % If we had removed, we would not need to update the focus because the next one would be "already there" 
            end
        end
        if i_restart
            i_restart = false;
        else
            i = i + 1;
        end
    end
end

function [spec_list, spec_radii, spec_values] = Shorten_Spec_List(spec_list, spec_radii, spec_values, max_spec_num)
    num_species = size(spec_list, 2); % The number of columns is equal to the length of the list of species
    if num_species > max_spec_num % We have too many species:
        num_to_kill = num_species - max_spec_num;
        [~ , indices] = sort(spec_radii, 'ascend'); % Higher level species (i.e., LOWER RADIUS) are deleted first
        spec_list(:, indices(1:1:num_to_kill)) = [];
        spec_radii(indices(1:1:num_to_kill)) = [];
        spec_values(indices(1:1:num_to_kill)) = [];
    end
end

function [spec_list, spec_values] = Optimize_Species(spec_list, spec_radii, spec_values, bounds, func, budget, local_optimizer, optimizer_config)
    for i=1:1:size(spec_list, 2) % The number of columns is equal to the length of the list of species
        [new_point, new_value] = local_optimizer(spec_list(:, i), spec_radii(i), spec_values(i), bounds, func, budget, optimizer_config);
        if new_value < spec_values(i) % If a better new center has been found, replace (but do not change the radius)
           spec_list(:, i) = new_point;
           spec_values(i) = new_value;
        end
    end
end
