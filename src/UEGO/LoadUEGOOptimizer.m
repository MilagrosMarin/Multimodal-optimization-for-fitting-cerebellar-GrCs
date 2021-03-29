function [local_optimizer, optim_config] = LoadUEGOOptimizer(name)
% LoadOptimizer  Returns a valid local optimizer for UEGO. It also loads
% the default configuration that the optimizer requires (when necessary).
    addpath('/home/nicolas/Documentos/SOFTWARE/Optimizadores/UEGO/LocalOptimizers/');
    if exist('name', 'var')
        switch name
            case 'SASS'
                [local_optimizer, optim_config] = SASS();
            case 'FMC'
                if hasToolbox('Optimization Toolbox')
                    [local_optimizer, optim_config] = FMC();
                else
                    disp('Warning: Optimization Toolbox not available for FMC. Using the default SASS instead!');
                    [local_optimizer, optim_config] = SASS();
                end
            otherwise
                disp('Warning: Unknown local optimizer. Loading SASS by default');
                [local_optimizer, optim_config] = SASS();
        end
    else
        disp('Warning: No local optimizer requested. Loading SASS by default');
        [local_optimizer, optim_config] = SASS();
    end
end

%----------------------- Internal auxiliary functions:

function available = hasToolbox(name)
    v = ver; % See: https://www.mathworks.com/matlabcentral/answers/37252-how-to-test-if-toolbox-exists
    available = any(strcmp(cellstr(char(v.Name)), name));
end
