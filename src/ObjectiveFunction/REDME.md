FF4 objective function by Marín et al., 2020 [1]
===========================================================

Background information
----------------------

This folder contains the objective function referred to as FF4 in [1] that has also been used in [2]. It is a Python procedure built upon the Nest simulator [3] and wrapped to be callable from Matlab. The model internally used is AdEx [4] adapted to Cerebellar Granule Cell (See [1] for further information).

The code in this folder has been used for the paper "On the use of a multimodal optimizer for fitting neuron models. Application to the cerebellar granule cell" by M. Marín, N.C. Cruz, E.M. Ortigosa, M.J. Saéz-Lara, J. Garrido and R.R. Carrillo. The referred paper has been submitted to *Frontiers in Cellular Neuroscience* [2]. The code is hence attached to the paper to support open science and simplify either replicating the study or using and extending the method.

The source code can be used and modified by anyone interested in doing so as long as the reference to the paper in *Frontiers* is included. Be warned, the code is provided as-is, with the best of intentions but any kind of warranty neither of correctness nor computational efficiency. None of the people mentioned above can be charged with any kind of responsibility concerning this code.

The underlying execution requirements for numerical stability are the following:

- Python 2.7.17

- Nest 2.14.0 

See [5] for an installation tutorial.

Example of use
--------------

Provided that the environment requirements have been successfully installed, let us see how to evaluate a particular model configuration or candidate solutions in optimization terms. 
First, our Matlab instance must be launched from a user-session that has loaded the environment variables of Nest:

```console
user@pc:~/Path/to/this/folder$ source ~/Path/to/Nest/SOFTWARE/Nest2_14/bin/nest_vars.sh 
user@pc:~/Path/to/this/folder$ /usr/local/MATLAB/R2018b/bin/matlab -desktop

```

From the resulting Matlab instance, we can evaluate the candidate solutions as described in [1, 2] as follows. Let us evaluate the best solution for FF4 in [1]:

```Matlab
% We first load the model configuration and define the simulation seed (which is independent of any used at Matlab level for optimization)

pyContext = NC_Load_Static_Context('./Inputs/NC_NeuronModel9PRes_seed1.cfg');
pyContext.seed= py.int(12345);	

% The, we create a vector with the desired configuration (PAY ATTENTION TO THE COMPONENTS):
% Ordering: # mflayer.a; mflayer.espike; mflayer.eth; mflayer.b; mflayer.cm; mflayer.erest; mflayer.grest; mflayer.delta_t; mflayer.tw; mflayer.vreset

individual = [0.000000000232126018536459164497,... 	% mflayer.a
	 -0.017561476076408549101826039873, ...		% mflayer.espike
	-0.024010273317557472017025332889, ...		% mflayer.eth
	0.000000000370715753949636493039, ...		% mflayer.b
	0.000000000002798385984637836598, ...		% mflayer.cm
	-0.058002923844557574550862000251, ...		% mflayer.erest
	0.000000000246025590829987885428, ...		% mflayer.grest
	0.022074048991742884623379339359, ...		% mflayer.delta_t
	0.619071345857195476369838615938, ...		% mflayer.tw
	-0.071314565810638108622754316457];		% mflayer.vreset

NC_ObjFunc(individual, pyContext)

ans =

  104.2358

% By the way, it does not matter if you work with a column or row vectors

NC_ObjFunc(individual', pyContext)

ans =

  104.2358

```

### References

[[1]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7381211/) M. Marín, M.J. Sáez-Lara, E. Ros, & J.A. Garrido. Optimization of Efficient Neuron Models With Realistic Firing Dynamics. The Case of the Cerebellar Granule Cell. *Frontiers in Cellular Neuroscience*, 14, 2020.

[2] M. Marín, N.C. Cruz, E.M. Ortigosa, M.J. Saéz-Lara, J. Garrido & R.R. Carrillo. On the use of a multimodal optimizer for fitting neuron models. Application to the cerebellar granule cell. Submitted to *Frontiers in Cellular Neuroscience*, 2021.

[[3]](https://www.nest-simulator.org/) A. Peyser et al. (2017). NEST 2.14.0. Zenodo. 10.5281/zenodo.882971

[[4]](https://journals.physiology.org/doi/full/10.1152/jn.00686.2005) R. Brette & W. Gerstner. Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. *Journal of Neurophysiology*, 94(5), 3637-3642, 2005.

[[5]](https://github.com/nest/nest-simulator/issues/866) Nest installation with python
