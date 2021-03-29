# Introduction
This repository is linked to the paper *"On the use of a multimodal optimizer for fitting neuron models. Application to the cerebellar granule cell"*, which has been submitted to *Frontiers in Neuroinformatics* and is authored by M. Marín, N.C. Cruz, E.M. Ortigosa, M.J. Sáez-Lara, J.A. Garrido and R.R. Carrillo. This repository contains the source code of the multimodal optimizer, its cost function, and the results described in the referred paper. Details such as the definition of the target problem, the proposed workflow and methodology used, as well as the resulting population of neuron models of cerebellar granule cells can be found in that paper. The reader is hence referred to that work for better understanding the information included herein.
<br/><br/>
# Version
Current version: 1.0 <28/03/21>
<br/><br/>
# Usage
- <u>"src" folder</u>: it contains the Matlab source code of the multimodal algorithm *UEGO* and the fitness function. Specifications about the execution of the algorithm are included in its README.md
- <u>"images" folder</u>: it contains the images generated for the paper. 
- <u>"results" folder</u>: it contains the dataset with the results of 10 independent executions of UEGO using different seeds. Execution 2 (E2) is the resulting population of neuron models described in the article (25 solutions). The capability of finding different yet similarly-ranked configurations is demonstrated by the figure 5B in the article (for execution 2) and in the Supplementary Material 1 (for the rest).
- <u>"notebook" folder</u>: it contains a jupyter notebook that simulates and analyses quantitatively and qualitatively the neuron spiking dynamics used as the fitness function. A file of some utility functions and a configuration file of the default values for some internal variables, both needed for the notebook execution, are also included. 

For any question or suggestion feel free to contact the corresponding author: M. Marín (mmarin AT ugr.es).
<br/><br/>
# License
This software is provided 'as-is', with the best intention but with any kind of warranty of responsibility regarding stability, efficiency and correctness. It can be freely used, distributed and modified by anyone interested at it, but it is REQUIRED to include the following reference:

- Marín, M., Cruz, N.C., Ortigosa, E.M., Sáez-Lara, M.J., Garrido, J.A. & Carrillo, R.R.(2021). On the use of a multimodal optimizer for fitting neuron models. Application to the cerebellar granule cell. *Frontiers in Neuroinformatics*. 
