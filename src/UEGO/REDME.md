Universal Evolutionary Global Optimizer (UEGO)
==============================================

Background information
----------------------

This folder contains a general-purpose Matlab implementation of the [Universal Evolutionary Global Optimizer (UEGO)](https://link.springer.com/content/pdf/10.1023/A:1011367930251.pdf) algorithm [1, 2]. 
It is a memetic [3] multimodal [4] optimization method. 
According to its memetic nature, UEGO is an evolutionary optimizer, but its candidate solutions are not passive as in standard genetic algorithm. 
Instead, they are active and try to autonomously improve through local optimization.
As a multimodal method, UEGO will try to identify and return as many optimal and suboptimal as possible. This kind of method is useful when: 

1. It is possible that some model-based solutions are either unfeasible or not so attractive in reality as it seemed for the model-based cost function (e.g., noise and inaccuracy), so it is necessary to post-process the results and ultimately select the most convenient solution.

2. The ultimate aim is to study the search space of the underlying problem, so obtaining an immediate set of different solutions in a single run is useful.

The interested reader can also find a gentle introduction and become familiar with the terms previously referred through the following Wikipedia articles:

- [Memetic algorithm](https://en.wikipedia.org/wiki/Memetic_algorithm)

- [Evolutionary multimodal optimization](https://en.wikipedia.org/wiki/Evolutionary_multimodal_optimization)

- [UEGO (Spanish only)](https://es.wikipedia.org/wiki/UEGO)

The main local search algorithm included is an ad-hoc implementation of Solis and Wets' method [6], also known as SASS (derived from Single-Agent Stochastic Search) [2]. It is a stochastic hill-climber with adaptive search. See [7] for a detailed explanation about how to implement the method.

The code in this folder has been used for the paper "On the use of a multimodal optimizer for fitting neuron models. Application to the cerebellar granule cell" by M. Marín, N.C. Cruz, E.M. Ortigosa, M.J. Saéz-Lara, J. Garrido and R.R. Carrillo. The referred paper has been submitted to *Frontiers in Cellular Neuroscience* [8]. The code is hence attached to the paper to support open science and simplify either replicating the study or using and extending the method.

The optimizer itself has been initially coded by N.C. Cruz (2019) under the supervision of J.L. Redondo, J.D. Alvarez, M. Berenguel and P.M. Ortigosa (University of Almería, Spain)

The source code can be used and modified by anyone interested in doing so as long as the reference to the paper in *Frontiers* is included. Be warned, the code is provided as-is, with the best of intentions but any kind of warranty neither of correctness nor computational efficiency. None of the people mentioned above can be charged with any kind of responsibility concerning this code.

The source code in this folder is as it was used in the paper by Marín et al, but it is detached from any kind of repository, so you will not benefit from any kind of support, bug fix or extension. Nevertheless, there is an active git repository for this Matlab implementation of UEGO through the following link: [UEGO](https://github.com/cnelmortimer/UEGO) (Not available until publication)

Example of use for the problem addressed in [8]
-----------------------------------------------

As introduced, the present optimizer implementation can be used to solve any continuous and unconstrained (box constraints are required, though) optimization problem. However, the code below shows how it has been configured to be used in [8]. See the referred paper to fully understand the parts described. Notice how the search space is externally normalized for numerical stability and because of the implementation of the local search method used.

```Matlab
% We assume that our Matlab is in the folder of the cost function properly linked
% So let us first help Matlab to load the UEGO methods:

addpath('/Path/to/the/folder/of/UEGO/');

% Let us define the problem bounds, i.e., a matrix with as many rows as variables and 2 columns
% The first column is for the lower bound, and the second one is for the upper bound
% The order definitely matters both in rows and columns! See how the cost function expects the
% components:

bounds = zeros(10, 2);
bounds(1,:) = [-0.000000001 0.000000001]; % a
bounds(2,:) = [-0.02 0.02]; % V_Peak (espike)
bounds(3,:) = [-0.06 -0.02]; % V_T (eth)
bounds(4,:) = [-0.000000001 0.000000001]; % b
bounds(5,:) = [0.0000000000001 0.000000000005]; % C_m (cm)
bounds(6,:) = [-0.08 -0.04]; % E_L (erest)
bounds(7,:) = [0.000000000001 0.00000001]; % g_L (grest)
bounds(8,:) = [0.001, 1]; % Delta_T (delta_t)
bounds(9,:) = [0.001, 1]; % T_w (tw)
bounds(10,:) = [-0.08 -0.04]; % V_r (vreset)

% But we will separate UEGO and its internal method from the real ranges, so let us define
% the ranges that the optimizer will see in reality:
normalBounds = zeros(10, 2);
normalBounds(:,2) = ones(10,1);

% The cost function needs to launch Nest with the desired configuration and seed, so load it:

pyContext = NC_Load_Static_Context('./Inputs/NC_NeuronModel9PRes_seed1.cfg');
pyContext.seed= py.int(12345);

% Let us define lambda function based the cost function which handles normalization and denormalization.
% Notice that we also provide it with the previous background information:

normalFunc = @(x) NC_NormalizedObjFunc(x, @NC_ObjFunc, pyContext, bounds);

% The normalized function also registers the number of calls, so prepare this information:

global numCalls;
numCalls = 0;

% We can now define the paramters expected by UEGO, i.e., maximum number of evaluations, number of levels,
% maximum number of species and minimum radius for the last level:

evals = 1000000; levels = 50; max_spec_num = 100; min_r = 0.7;

% It is also necessary to load one of the local search methods provided by UEGO, SASS in this case.
% DO NOT FORGET to modify the addpath included in LoadUEGOOptimizer to point to the LocalOptimizers sub-folder in UEGO/:

[local_optimizer, optim_config] = LoadUEGOOptimizer('SASS');

% Let's go: Launch the method and wait

tA = tic;
[config, spec_list, spec_radii, spec_values] = UEGO(evals, levels, max_spec_num, min_r, normalBounds, normalFunc, local_optimizer, optim_config);
time = toc(tA);

% The outputs are: 
% 	The internal UEGO configuration for debugging purposes
%	The final population of UEGO, i.e., the sets of different results or candidate solutions (There is a column for every solution with a row for each variable)
%	The radius of each species for debugging purposes
%	The fitness of each species in the final population, i.e., the result of the objective function for the corresponding solution
	
```

### References

[[1]](https://link.springer.com/content/pdf/10.1023/A:1011367930251.pdf) M. Jelasity, P.M. Ortigosa & I. García. UEGO, an abstract clustering technique for multimodal global optimization. *Journal of Heuristics*, 7(3), 215-233, 2001.

[[2]](https://link.springer.com/content/pdf/10.1023/A:1011224107143.pdf) P.M. Ortigosa, I. García & M. Jelasity. Reliability and performance of UEGO, a clustering-based global optimizer. *Journal of Global Optimization*, 19(3), 265-289, 2001.

[[3]](https://www.researchgate.net/profile/Pablo-Moscato/publication/2354457_On_Evolution_Search_Optimization_Genetic_Algorithms_and_Martial_Arts_-_Towards_Memetic_Algorithms/links/54b32b950cf220c63cd27988/On-Evolution-Search-Optimization-Genetic-Algorithms-and-Martial-Arts-Towards-Memetic-Algorithms.pdf) P. Moscato. On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms. *Caltech concurrent computation program*, C3P Report, 826, 1989.

[[4]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=735432) B. Sareni & L. Krahenbuhl. Fitness sharing and niching methods revisited. *IEEE transactions on Evolutionary Computation*, 2(3), 97-106, 1998.

[[5]](https://watermark.silverchair.com/evco.2010.18.1.18104.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAApgwggKUBgkqhkiG9w0BBwagggKFMIICgQIBADCCAnoGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMewO1SalRc_s8Ns9OAgEQgIICS9_ZEMKov9nWUNnmp2zJH4jVhzZ0UkZ-zVPIbDs7aqDRVuDz2OR0IJiDccIhOP6K9DN7GNijjvxpk5mtlj43Yyz1HC6xXt62v9ufpnmNi4q4M0MT4KC3lLR1815WzIXeK-u4KpLdVS8asZUMvtWlWY4-vAk1BFhbX81_wc5vWU42tYGAb9iaxcFflziueGoCNUbQJW9mwBnRaZ-cS6fxLMLPB1ZH69QuI-0pKF2b_aOCNCdpvWbnAqN401UqX_LIxPyxfBc0gmzb2po8RJ8sBgfbcIZot3NnetO2vJu_WOHOuaYeQTXLGbVQBKVb_35CLP0LNfxjUVzHvixImHDAFYeAt9qHcDrAC_zpYj8gqyI2IVA6BYHheuHBs26O2Is_YrxBDXq2AAnHGH7bK2noWh_O9MKGUkA-hOo_omLF62uHtXWeimj4KW2NAxj_0-XApddxK1rRsfEv53J2dDFVc7WYerK-gKzeBSd8WRtVoq6fbgqVvw6pSI3fnLu1-DkSTcCRtej98hYbdRLy_jRiFbth9_JoFwJwfddvMdmDNihFPHzmp2isdAatjn6IrCK0U-p9HIG5C926N7RTZhunEmESAv5zEeykdRzThA4FUIcfzdHB8Ip7C-L03zqWPKzF6B8CzcgMBFUC_XGTbFZQVf5k4ygeIzacWN_M5VxgcClcDtLbCszoEozkRQnS2lokDWJwowjo5JjJmO6ygHRQeZLlrbd7Ey_-MzBQLsiO66op5EkMsScPuqeaSZ33PA9JM11dAWxkCjMQcSi9) O.M. Shir, M. Emmerich & T. Bäck. Adaptive niche radii and niche shapes approaches for niching with the CMA-ES. *Evolutionary Computation*, 18(1), 97-126, 2010.

[[6]](https://www.jstor.org/stable/pdf/3689263.pdf) F.J. Solis & R.J.B. Wets. Minimization by random search techniques. *Mathematics of Operations Research*, 6(1), 19-30, 1981.

[[7]](https://www.zurnalai.vu.lt/nonlinear-analysis/article/download/14011/12927) A. Lančinskas, P.M. Ortigosa & J. Žilinskas. Multi-objective single agent stochastic search in non-dominated sorting genetic algorithm. Nonlinear Analysis: Modelling and Control, 18(3), 293-313, 2013.

[8] M. Marín, N.C. Cruz, E.M. Ortigosa, M.J. Saéz-Lara, J. Garrido & R.R. Carrillo. On the use of a multimodal optimizer for fitting neuron models. Application to the cerebellar granule cell. Submitted to *Frontiers in Cellular Neuroscience*, 2021.
