'''
Created on April 23, 2014

@author: Jesus Garrido (jgarridoalcazar at gmail.com)
'''
import numpy
import time
import logging
import scipy.spatial.distance

logger = logging.getLogger('Simulation')

class SynapticLayer(object):
    '''
    This class defines a synaptic layer and the data needed to generate it in a simulator.
    If several MPI processes are used, each process store only the loacl connections (those
    reaching a local cell).
    '''
    
    # Memory block size. It determines how the connectivity matrixes will be splitted. It should be chosen according to the amount of memory in the computer.
    memory_block =  32*1024*1024 # 32M of elements
    
    def __init__(self,**kwargs):
        '''
        Constructor of the class. It creates a new synaptic layer.
        @param source_layer: Neuron/input source layer.
        @param target_layer: Neuron target layer.
        @param intermediate_layer: Third layer to be used to create the connections (i.e. GoC-GrC connections). (Optional)
        @param intermediate_to_target_synaptic_layer: Synaptic layer connection the intermediate_layer and the target layer. (Optional)
        @param weight_recording: Whether the weights must be recorded.
        @param weight_normalization: Whether the weights must be normalizated.
        @param weight_sum: Total sum of the incoming connection weights for each target cell.
        @param random_generator: The random number generator to use.
        @param synaptic_delay: Transmission delay in the synapses (in seconds).
        @param connectivity_type: Pattern of connectivity in the connection. Only those patterns included in template_connectivity_parameters are allowed.
        @param connectivity_parameters: Connectivity algorithm parameters as described in the template_connectivity_parameters dictionary.
        @param weight_initialization_type: Weight initialization type in the connection. Only those patterns included in template_weight_parameters are allowed.
        @param weight_initialization_parameters: Weight initialization parameters as described in the template_weight_parameters dictionary.
        @param learning_rule_type: Learning rule type associated to the connections. Only those rules included in template_rule_parameters are allowed.
        @param learning_rule_parameters: Learning rule parameters as described in the template_rule_parameters dictionary.
        @param load_from_file: h5py group from which the synaptic layer connections and weights have to be loaded (optional).
        '''
        
        # Specific parameters for each learning rule
        self.template_rule_parameters = {
                         'STDP': ['tau_plus','learning_step','minus_plus_ratio','max_weight'],
                         'STDPSym': ['tau_sym','learning_step','minus_plus_ratio','max_weight'],
                         'eSTDP': ['tau_plus','learning_step','minus_plus_ratio','max_weight'],
                         'iSTDP': ['tau_plus','learning_step','minus_plus_ratio','max_weight'],
                         'STDPTriplet': ['tau_plus','tau_plus_triplet','learning_step','learning_step_triplet','minus_plus_ratio','minus_plus_ratio_triplet','max_weight']
                         }
    
        # Specific parameters of the connectivity algorithm (and function implementing each one)
        self.template_connectivity_parameters = {
                                        'fixedn2one': ['number_of_source_cells', self._generate_fixed_n2one_connections_],
                                        'randomn2one': ['number_of_source_cells', self._generate_random_n2one_connections_],
                                        'randomn2onestd': ['average_number_of_source_cells', 'std_number_of_source_cells', self._generate_random_n2one_std_connections_],
                                        'random_with_probability': ['connection_probability', self._generate_random_connections_],
                                        'randomn2onelocal': ['average_number_of_source_cells', 'average_dendritic_length', self._generate_random_n2one_local_connections_],
                                        'randomn2onestdlocal': ['average_number_of_source_cells', 'std_number_of_source_cells', 'average_dendritic_length', 'std_dendritic_length', self._generate_random_n2one_std_local_connections_],
                                        'glomerulin2onelocal': ['average_number_of_source_cells_per_glomerulus', 'average_dendritic_length', self._generate_glomeruli_n2one_local_connections_]
                                        }

        # Specific parameters of the weight initialization (and function implementing each one)
        self.template_weight_parameters = {
                                        'random': ['random_min_weight', 'random_max_weight', self._generate_random_weights_],
                                        'fixed': ['initial_weight', self._generate_fixed_weights_]
                                        }
        # Read name
        self.__name__ = kwargs.pop('name')
        
        # Read source layer
        if ('source_layer' in kwargs):
            self.source_layer = kwargs.pop('source_layer')
        else:
            logger.error('Non-specified source layer')
            raise Exception('Non-DefinedProperty')
        
        # Read target layer
        if ('target_layer' in kwargs):
            self.target_layer = kwargs.pop('target_layer')
        else:
            logger.error('Non-specified target layer')
            raise Exception('Non-DefinedProperty')
        
        # Read intermediate layer
        if ('intermediate_layer' in kwargs):
            self.intermediate_layer = kwargs.pop('intermediate_layer')
        else:
            self.intermediate_layer = None
            
        # Read intermediate_to_target_ synaptic layer
        if ('intermediate_to_target_synaptic_layer' in kwargs):
            self.intermediate_to_target_synaptic_layer = kwargs.pop('intermediate_to_target_synaptic_layer')
        else:
            self.intermediate_to_target_synaptic_layer = None
        
        # Read seed
        if ('random_generator' in kwargs):
            self.random_generator = kwargs.pop('random_generator')
        else:
            logger.warning('Non-specified random generator for synaptic layer generation. Using a new one')
            self.random_generator = numpy.random.RandomState(time())
        
        # Read synaptic type
        if ('weight_recording' in kwargs):
            self.weight_recording = kwargs.pop('weight_recording')
        else:
            self.weight_recording = False
            
        self.weight_record = dict()
        
        # Read synaptic type
        if ('weight_normalization' in kwargs):
            self.weight_normalization = kwargs.pop('weight_normalization')
            
            if self.weight_normalization:
                if ('weight_sum' in kwargs):
                    self.weight_total_sum = kwargs.pop('weight_sum')
                else:
                    logger.warning('Non-specified weight sum for normalization. Using default value 1e-9')
                    self.weight_total_sum = 1.e-9            
        else:
            self.weight_normalization = False
            
        # Read synaptic delay
        if ('synaptic_delay' in kwargs):
            self.synaptic_delay = kwargs.pop('synaptic_delay')
        else:
            logger.error('Non-specified synaptic delay')
            raise Exception('Non-DefinedProperty')
        
        # Read if the synaptic connections have to be loaded from file
        if ('load_from_file' in kwargs):
            self.load_from_file = kwargs.pop('load_from_file')
        else:
            self.load_from_file = None
        
        # Read connectivity type and its properties
        if ('connectivity_type' in kwargs):
            self.connectivity_type = kwargs.pop('connectivity_type')
            if (self.connectivity_type in self.template_connectivity_parameters):
                self.connectivity_parameters = {}
                for param in self.template_connectivity_parameters[self.connectivity_type][:-1]:
                    if (param in kwargs):
                        self.connectivity_parameters[param] = kwargs.pop(param)
                    else:
                        logger.warning('Non-specified connectivity parameter: %s in layer %s. Using default value', param, self.__name__)
                
                if 'allow_auto_connection' in kwargs:
                    self.connectivity_parameters['allow_auto_connection'] = kwargs.pop('allow_auto_connection')
                else:
                    self.connectivity_parameters['allow_auto_connection'] = True
                    
                if 'allow_multiple_connections' in kwargs:
                    self.connectivity_parameters['allow_multiple_connections'] = kwargs.pop('allow_multiple_connections')
                else:
                    self.connectivity_parameters['allow_multiple_connections'] = True
                
                if (self.load_from_file is None):
                    # Generate the individual connections
                    self.template_connectivity_parameters[self.connectivity_type][-1]()
                else:
                    self.load_layer(self.load_from_file)
            else:
                logger.error('Unknown connectivity type: %s', self.connectivity_type)
                raise Exception('UnknownConnectivityType')                         
        else:
            logger.error('Non-specified connectivity type')
            raise Exception('Non-DefinedProperty')
        
        # Read weight initialization and its properties
        if ('weight_initialization_type' in kwargs):
            self.weight_initialization_type = kwargs.pop('weight_initialization_type')
            if (self.weight_initialization_type in self.template_weight_parameters):
                self.weight_initialization_parameters = {}
                for param in self.template_weight_parameters[self.weight_initialization_type][:-1]:
                    if (param in kwargs):
                        self.weight_initialization_parameters[param] = kwargs.pop(param)
                    else:
                        logger.error('Non-specified weight initialization parameter: %s',param)
                        raise Exception('Non-DefinedWeightInitializationParameter')
                # Generate the individual connections
                if (self.load_from_file is None):
                    # Generate the synaptic weights
                    self.template_weight_parameters[self.weight_initialization_type][-1]()
            else:
                logger.error('Unknown weight initialization type: %s', self.weight_initialization_type)
                raise Exception('UnknownWeightInitializationType')                         
        else:
            logger.error('Non-specified weight initialization type')
            raise Exception('Non-DefinedProperty')
        
        # Read learning rule and its properties
        if ('learning_rule_type' in kwargs):
            self.learning_rule_type = kwargs.pop('learning_rule_type')
            if (self.learning_rule_type in self.template_rule_parameters):
                self.learning_rule_parameters = {}
                for param in self.template_rule_parameters[self.learning_rule_type]:
                    if (param in kwargs):
                        self.learning_rule_parameters[param] = kwargs.pop(param)
                    else:
                        logger.warning('Non-specified learning rule parameter: %s in layer %s. Using default value', param, self.__name__)
            else:
                logger.error('Unknown learning rule type: %s', self.learning_rule_type)
                raise Exception('UnknownLearningRuleType')                         
        else:
            self.learning_rule_type = None
            
        # Gather the synapses to every node
        # It is not efficient in terms of memory consumption, but it makes this class independent of NEST simulator
        #self._share_connections()
        
        # Check whether additional parameters have been used.
        for param in kwargs:
            logger.warning('Unrecognized parameter %s in layer %s',param,self.__name__)
            
        return
    
    def _generate_fixed_n2one_connections_(self):
        '''
        Generate connections between source and target layers n2one following succesive order (s0,s1,...,s(n-1) to t0, sn, s(n+1),...,s(2n-1) to t1,...).
        '''
        if (self.source_layer.number_of_neurons != self.target_layer.number_of_neurons * self.connectivity_parameters['number_of_source_cells']):
            logger.error('Invalid connectivity type. Check source and target layer neuron number')
            raise Exception('InvalidPropertyValue') 
         
        # Get the number of parallel proecesses
        size = 1
        rank = 0
         
        # Calculate the number of synapses this process generates dividing the number of target cells between the number of processes
        local_target_cells, = numpy.where(self.target_layer.is_local_node)        
         
        self.number_of_synapses = len(local_target_cells) * self.connectivity_parameters['number_of_source_cells']
         
        # Generate source cell indexes of the connections
        self.source_index = numpy.concatenate([range(self.connectivity_parameters['number_of_source_cells']*index,self.connectivity_parameters['number_of_source_cells']*(index+1)) for index in local_target_cells]).astype(numpy.uint32)
        self.target_index = numpy.repeat(local_target_cells,self.connectivity_parameters['number_of_source_cells']).tolist().astype(numpy.uint32)
        return
     
    def _generate_random_n2one_connections_(self):
        '''
        Generate connections between source and target layers n2one with random selection of source cells (sr0,sr1,...,sr(n-1) to t0, srn, s(rn+1),...,s(r2n-1) to t1,...).
        '''
        # Get the number of parallel proecesses
        size = 1
        rank = 0
         
        num_source_cells = self.connectivity_parameters['number_of_source_cells']

        # Calculate the number of synapses this process generates dividing the number of target cells between the number of processes
        local_target_cells, = numpy.where(self.target_layer.is_local_node)
        self.number_of_synapses = len(local_target_cells) * num_source_cells
         
        # For each target cell generate a permutation of the source indexes
        self.source_index = numpy.zeros(self.number_of_synapses,dtype = numpy.uint32)
        for i in range(len(local_target_cells)):
            self.source_index[i*num_source_cells:(i+1)*num_source_cells] = self.random_generator.permutation(numpy.arange(self.source_layer.number_of_neurons))[0:num_source_cells].astype(numpy.uint32)
        self.target_index = numpy.repeat(local_target_cells,num_source_cells).astype(numpy.uint32)
         
        return

    def _generate_random_n2one_std_connections_(self):
        '''
        Generate connections between source and target layers n2one with random selection of source cells (sr0,sr1,...,sr(n-1) to t0, srn, s(rn+1),...,s(r2n-1) to t1,...).
        '''
        # Get the number of parallel proecesses
        size = 1
        rank = 0
         
        # Calculate the number of synapses this process generates dividing the number of target cells between the number of processes
        local_target_cells, = numpy.where(self.target_layer.is_local_node)

        mean_num_source = self.connectivity_parameters['average_number_of_source_cells']
        std_num_source = self.connectivity_parameters['std_number_of_source_cells']

        # Generate the number of source cells for each target cells
        num_source_cells = numpy.rint(self.random_generator.normal(mean_num_source, std_num_source, len(local_target_cells))).astype(numpy.uint32)
        num_source_cells[num_source_cells<0] = 0
        self.number_of_synapses = numpy.sum(num_source_cells)
        
        # For each target cell generate a permutation of the source indexes
        self.source_index = numpy.empty(self.number_of_synapses, dtype=numpy.uint32)
        self.target_index = numpy.empty(self.number_of_synapses, dtype=numpy.uint32)
        init_index = 0
        end_index = 0
        for num_source, num_target in zip(num_source_cells, local_target_cells):
            init_index = end_index
            end_index = init_index + num_source
            if (self.connectivity_parameters['allow_multiple_connections']):
                self.source_index[init_index:end_index] = self.random_generator.randint(self.source_layer.number_of_neurons,size=num_source).astype(numpy.uint32)
            else:
                self.source_index[init_index:end_index] = self.random_generator.permutation(numpy.arange(self.source_layer.number_of_neurons))[0:num_source].astype(numpy.uint32)
            self.target_index[init_index:end_index] = numpy.repeat(num_target,num_source).astype(numpy.uint32)
         
        return
     
    def _generate_random_connections_(self):
        '''
        Generate connections between source and target layers with a given probability.
        '''
        # Get the number of parallel proecesses
        size = 1
        rank = 0
         
        probability = self.connectivity_parameters['connection_probability']
 
        # Calculate the number of synapses this process generates dividing the number of target cells between the number of processes
        local_target_cells, = numpy.where(self.target_layer.is_local_node)
        
        # Number of target cells in each memory block
        target_layers_in_block = self.memory_block / self.source_layer.number_of_neurons
        
        # Distribute the local target cells in the number of blocks
        local_target_indexes = range(0,len(local_target_cells), target_layers_in_block)
        
        self.source_index = numpy.array([], dtype=numpy.uint32)
        self.target_index = numpy.array([], dtype=numpy.uint32)
        
        for i,_ in enumerate(local_target_indexes):
            if i==(len(local_target_indexes)-1):
                target_cells = local_target_cells[local_target_indexes[i]:]
            else:
                target_cells = local_target_cells[local_target_indexes[i]:local_target_indexes[i+1]]  
        
            # Generate num_source_cells * num_target_cells rand numbers
            rand_numbers = self.random_generator.rand(self.source_layer.number_of_neurons,len(target_cells))
     
            # Autoconnections are not allowed
            if self.source_layer == self.target_layer:
                for index in range(len(target_cells)):
                    rand_numbers[index,index] = 1.0
     
            (source_array,target_array) = numpy.where(rand_numbers<probability)
            self.source_index = numpy.append(self.source_index,source_array.astype(numpy.uint32))
            self.target_index = numpy.append(self.target_index, target_cells[target_array].astype(numpy.uint32))
            
            logger.debug('Generated connections in layer %s (%s of %s). Local: %s', self.__name__,i+1,len(local_target_indexes),len(target_array))  
    
        self.number_of_synapses = self.target_index.shape[0]      
        return
    
    def _generate_random_n2one_local_connections_(self):
        '''
        Generate connections between source and target layers n2one with random selection of source cells (sr0,sr1,...,sr(n-1) to t0, srn, s(rn+1),...,s(r2n-1) to t1,...).
        In this case the probability of connections is generated according to an exponential distribution with average length the average_dendritic_length parameter.
        '''
        
        target_coord = self.target_layer.get_absolute_coordinates()
        source_coord = self.source_layer.get_absolute_coordinates()
        
        n_source_cells = source_coord.shape[0]
        n_target_cells = target_coord.shape[0]
        
        # Calculate the number of synapses this process generates dividing the number of target cells between the number of processes
        local_target_cells, = numpy.where(self.target_layer.is_local_node)
        
        lambda_val = 1./self.connectivity_parameters['average_dendritic_length']
        
        # Number of target cells in each memory block
        target_layers_in_block = self.memory_block / self.source_layer.number_of_neurons
        
        # Distribute the local target cells in the number of blocks
        local_target_indexes = range(0,len(local_target_cells), target_layers_in_block)
        
        self.source_index = numpy.array([], dtype=numpy.uint32)
        self.target_index = numpy.array([], dtype=numpy.uint32)
        
        for i,_ in enumerate(local_target_indexes):
            if i==(len(local_target_indexes)-1):
                target_cells = local_target_cells[local_target_indexes[i]:]
            else:
                target_cells = local_target_cells[local_target_indexes[i]:local_target_indexes[i+1]]  
        
            distance = scipy.spatial.distance.cdist(source_coord, target_coord[target_cells], 'euclidean')
            
            # Generate the probability of connection for each pair of source/target neurons following an exponential distribution with average_dendritic_length
            probability = lambda_val * numpy.exp(-lambda_val*distance)
            # For each target neuron normalize the probability and multiply by the average number of source cells
            norm_probability = probability / numpy.sum(probability,axis=0) * self.connectivity_parameters['average_number_of_source_cells']
            
            # Generate random numbers according to uniform distribution and compare with the probability of each connection
            rand_nums = self.random_generator.uniform(0,1,probability.shape)
     
            # Autoconnections are not allowed
            if self.source_layer == self.target_layer:
                for index in range(len(target_cells)):
                    rand_nums[index,index] = 1.0
     
            (source_array,target_array) = numpy.where(rand_nums<norm_probability)
            self.source_index = numpy.append(self.source_index,source_array.astype(numpy.uint32))
            self.target_index = numpy.append(self.target_index, target_cells[target_array].astype(numpy.uint32))
            
            logger.debug('Generated connections in layer %s (%s of %s). Local: %s', self.__name__,i+1,len(local_target_indexes),len(target_array))
        
        self.number_of_synapses = self.target_index.shape[0]
        return
    
    def _generate_random_n2one_std_local_connections_(self):
        '''
        Generate connections between source and target layers n2one with random selection of source cells (sr0,sr1,...,sr(n-1) to t0, srn, s(rn+1),...,s(r2n-1) to t1,...).
        In this case the number of connections is generated according to a normal distribution (with mean and std indicated) and the probability of connection is
        generated according to the distance between source and target cell (with normal distribution as indicated).
        '''
        
        import scipy.stats
        
        target_coord = self.target_layer.get_absolute_coordinates()
        source_coord = self.source_layer.get_absolute_coordinates()
        
        n_source_cells = source_coord.shape[0]
        n_target_cells = target_coord.shape[0]
        
        # Calculate the number of synapses this process generates dividing the number of target cells between the number of processes
        local_target_cells, = numpy.where(self.target_layer.is_local_node)
        
        mean_num_source = self.connectivity_parameters['average_number_of_source_cells']
        std_num_source = self.connectivity_parameters['std_number_of_source_cells']
        mean_dendritic_length = self.connectivity_parameters['average_dendritic_length']
        std_dendritic_length  = self.connectivity_parameters['std_dendritic_length']
        
        # Number of target cells in each memory block
        target_layers_in_block = self.memory_block / self.source_layer.number_of_neurons
        
        # Distribute the local target cells in the number of blocks
        local_target_indexes = range(0,len(local_target_cells), target_layers_in_block)
        
        self.source_index = numpy.array([], dtype=numpy.uint32)
        self.target_index = numpy.array([], dtype=numpy.uint32)
        
        for i,_ in enumerate(local_target_indexes):
            if i==(len(local_target_indexes)-1):
                target_cells = local_target_cells[local_target_indexes[i]:]
            else:
                target_cells = local_target_cells[local_target_indexes[i]:local_target_indexes[i+1]]  
        
            distance = scipy.spatial.distance.cdist(source_coord, target_coord[target_cells], 'euclidean')
            
            # Calculate the probability distribution from the distance and normalize
            pdf_distance = scipy.stats.norm.pdf(distance, mean_dendritic_length, std_dendritic_length)
            column_sums = pdf_distance.sum(axis=0)
            norm_pdf_distance = pdf_distance / column_sums[numpy.newaxis, :]
            cdf_distance = numpy.cumsum(norm_pdf_distance, axis=0)
            
            # Generate the number of source cells for each target cells
            num_source_cells = numpy.rint(self.random_generator.normal(mean_num_source, std_num_source, target_cells.shape[0]))
            
            source_block_list = []
            target_block_list = []
            
            for index, target_cell in enumerate(target_cells):
                source_list = []
                while len(source_list)<num_source_cells[index]:
                    rand_num = self.random_generator.uniform()
                    source_cell = numpy.argmax(cdf_distance[:,index]>rand_num)
                    pdf_distance[source_cell,index] = 0.0
                    if source_cell not in source_list:
                        source_list.append(source_cell)
                    else:
                        # Renormalize the probability matrix to avoid long-lasting looping
                        column_sum = pdf_distance[:,index].sum(axis=0)
                        norm_pdf_distance = pdf_distance[:,index] / column_sum
                        cdf_distance[:,index] = numpy.cumsum(norm_pdf_distance, axis=0) 
            
                source_block_list.extend(source_list)
                target_block_list.extend([target_cell]*len(source_list))
                
            self.source_index = numpy.append(self.source_index,source_block_list).astype(numpy.uint32)
            self.target_index = numpy.append(self.target_index,target_block_list).astype(numpy.uint32)
            
            logger.debug('Generated connections in layer %s (%s of %s). Local: %s', self.__name__,i+1,len(local_target_indexes),len(target_block_list))
        
        self.number_of_synapses = self.target_index.shape[0]
        return
    
    def _generate_glomeruli_n2one_local_connections_(self):
        '''
        Generate connections between source and target layers n2one with random selection of source cells (sr0,sr1,...,sr(n-1) to t0, srn, s(rn+1),...,s(r2n-1) to t1,...).
        In this case the probability of connections is generated according to an exponential distribution with average length the average_dendritic_length parameter.
        '''
        
        if self.intermediate_layer is None:
            logger.error('Non-specified intermediate layer. It is required in glomeruli-type connectivity creation.')
            raise Exception('Non-DefinedProperty')
        
        if self.intermediate_to_target_synaptic_layer is None:
            logger.error('Non-specified intermediate_to_target synaptic layer. It is required in glomeruli-type connectivity creation.')
            raise Exception('Non-DefinedProperty')
        
        
        # It is assumed that every neuron layer stores the coordinates of all the neurons (not only the local ones).
        interm_coord = self.intermediate_layer.get_absolute_coordinates()
        source_coord = self.source_layer.get_absolute_coordinates()
        
        n_source_cells = source_coord.shape[0]
        n_interm_cells = interm_coord.shape[0]
        
        lambda_val = 1./self.connectivity_parameters['average_dendritic_length']
                
        glomerulus_connection_index = []
        goc_connection_index = []
        
        # Generate the look-up table of target grcs and neighbouring glomeruli
        target_grcs_list = []
        neighbouring_glomeruli_list = []
        
        logger.debug('Creating glomerulus neighbourhood in layer %s. %s glomeruli', self.__name__, n_interm_cells)
        
        # ------------------------------------------------------------------------------------------------------------------
        # Collect all the intermediate_to_target_synaptic_layer (source_index and target_index in all the MPI processes)
        
        # intermediate_to_target_source_indexes -> Array with all the source indexes in the connections Glomeruli -> GrC
        # intermediate_to_target_target_indexes -> Array with all the target indexes in the connections Glomeruli -> GrC
        # intermediate_to_target_synaptic_count -> Number of synapses in the connections Glomeruli -> GrC
        intermediate_to_target_source_indexes = self.intermediate_to_target_synaptic_layer.source_index
        intermediate_to_target_target_indexes = self.intermediate_to_target_synaptic_layer.target_index
        intermediate_to_target_synaptic_count = self.intermediate_to_target_synaptic_layer.number_of_synapses
        
        # Sort the intermediate_to_target_synaptic_layer (target_indexes) according to the source layer
        target_order = intermediate_to_target_source_indexes.argsort()
        unique_source,split_indices = numpy.unique(intermediate_to_target_source_indexes[target_order], return_index=True)
        target_ordered = numpy.split(intermediate_to_target_target_indexes[target_order], split_indices[1:])
        
        target_all_ordered = []
        source_index = 0
        for i in range(self.intermediate_layer.number_of_neurons):
            if unique_source[source_index]==i:
                target_all_ordered.append(target_ordered[source_index])
                source_index = source_index + 1
            else:
                target_all_ordered.append(numpy.empty([0]))
                
        target_ordered = numpy.array(target_all_ordered)
        # target_ordered[ind] returns the target elements of the source glomeruli ind
                
        
        # Sort the intermediate_to_target_synaptic_layer (source_indexes) according to the target layer
        source_order = intermediate_to_target_target_indexes.argsort()
        unique_target,split_indices = numpy.unique(intermediate_to_target_target_indexes[source_order], return_index=True)
        source_ordered = numpy.split(intermediate_to_target_source_indexes[source_order], split_indices[1:])
        
        source_all_ordered = []
        target_index = 0
        for i in range(self.target_layer.number_of_neurons):
            if unique_target[target_index]==i:
                source_all_ordered.append(source_ordered[target_index])
                target_index = target_index + 1
            else:
                source_all_ordered.append(numpy.empty([0]))
                
        source_ordered = numpy.array(source_all_ordered)
        # source_ordered[ind] returns the source glomeruli of the target neuron ind
        
        # -------------------------------------------------------------------------------------------------------
        # This section has to be executed only in the first process when run with MPI
        # Number of interm cells in each memory block
        interm_layers_in_block = self.memory_block / self.source_layer.number_of_neurons
        
        # Distribute the intermediate cells in the number of blocks
        interm_indexes = range(0,n_interm_cells, interm_layers_in_block)
        
        for i,_ in enumerate(interm_indexes):
            if i==(len(interm_indexes)-1):
                glomerulus_indexes_block = range(interm_indexes[i], n_interm_cells)
            else:
                glomerulus_indexes_block = range(interm_indexes[i], interm_indexes[i+1])
                
                
            distance = scipy.spatial.distance.cdist(source_coord, interm_coord[glomerulus_indexes_block], 'euclidean')
        
            # Generate the probability of connection for each pair of source/target neurons following an exponential distribution with average_dendritic_length
            probability = lambda_val * numpy.exp(-lambda_val*distance)
        
            # Generate random numbers according to uniform distribution and compare with the probability of each connection
            rand_nums = self.random_generator.uniform(0,1,probability.shape)
            
            logger.debug('Calculating  neighbourhood in layer %s and block %s.', self.__name__, i+1)
            
            # Generate the neighbouring glomeruli
            if glomerulus_connection_index:
                target_cells = target_ordered[glomerulus_connection_index]
            else:
                target_cells = numpy.array([],dtype=numpy.uint32)
            
            for index_glom, grc_array in enumerate(target_cells):
                neighbouring_glomeruli = numpy.unique(numpy.hstack(source_ordered[grc_array]))
            
                current_glomeruli = numpy.logical_and(neighbouring_glomeruli>=glomerulus_indexes_block[0], neighbouring_glomeruli<=glomerulus_indexes_block[-1])
                
                adjusted_neighbour_indexes = numpy.array(neighbouring_glomeruli[current_glomeruli]) - glomerulus_indexes_block[0]
                
                goc_index_repeated = numpy.repeat([goc_connection_index[index_glom]],adjusted_neighbour_indexes.shape[0])
                
                probability[goc_index_repeated,adjusted_neighbour_indexes] = 0.0            
            
            logger.debug('Calculating  glomerulus connections in layer %s and block %s. %s glomeruli', self.__name__, i+1, len(glomerulus_indexes_block))
            # For each target neuron normalize the probability and multiply by the average number of source cells
            # When parallelized this part has to be done sequentially in the first process
            for glomerulus_index in glomerulus_indexes_block:
                local_index = glomerulus_index-glomerulus_indexes_block[0]
                
                norm_probability = probability[:,local_index] / numpy.sum(probability[:,local_index]) * self.connectivity_parameters['average_number_of_source_cells_per_glomerulus']
            
                connected_goc_indexes, = numpy.where(rand_nums[:,local_index]<norm_probability[:])
            
                # Store the GoC-Glomerulus connections
                glomerulus_connection_index.extend([glomerulus_index]*connected_goc_indexes.shape[0])
                goc_connection_index.extend(connected_goc_indexes)
            
                neighbour_glomeruli = numpy.unique(numpy.hstack(source_ordered[target_ordered[glomerulus_index]]))
                current_glomeruli = numpy.logical_and(neighbour_glomeruli>=glomerulus_indexes_block[0],neighbour_glomeruli<=glomerulus_indexes_block[-1])
                
                adjusted_neighbour_indexes = numpy.array(neighbour_glomeruli[current_glomeruli]) - glomerulus_indexes_block[0]
                
                # Update the probability of the following glomeruli to avoid repeating connections
                goc_indexes = numpy.tile(connected_goc_indexes,adjusted_neighbour_indexes.shape[0])
                glomeruli_rep_indexes = numpy.repeat(adjusted_neighbour_indexes,connected_goc_indexes.shape[0])
                probability[goc_indexes,glomeruli_rep_indexes] = 0.0
                
            logger.debug('Generated glomerulus-like connections in layer %s (%s of %s). Local: %s', self.__name__,i+1,len(interm_indexes),len(goc_connection_index))
        
        
        # ------------------------------------------------------------------------------------------------------------------
        # Send the GoC-Glomerulus connections lists to all the processes (each one will take only those connections reaching their local neurons)
        
        # Generate the GoC-GrC connections from the GoC-Glomerulus connections
        self.source_index = []
        self.target_index = []
        
        # Generate the target GrCs of the selected glomeruli
        target_grcs = target_ordered[glomerulus_connection_index]
        for i, target_group in enumerate(target_grcs):
            
            # Select the target grcs which are local to the current processor
            selected_target = self.target_layer.is_local_node[target_group] 
            self.source_index.extend(numpy.repeat(goc_connection_index[i],numpy.count_nonzero(selected_target)))
            self.target_index.extend(target_group[selected_target])
        
        logger.debug('Generated glomerulus-like connections in layer %s. Local: %s', self.__name__,len(self.source_index))
        
        self.source_index = numpy.array(self.source_index, dtype=numpy.uint32)
        self.target_index = numpy.array(self.target_index, dtype=numpy.uint32)
        self.number_of_synapses = self.target_index.shape[0]
        return
    
    def _generate_random_weights_(self):
        '''
        Generate uniformly distributed random weights between the minimum and maximum values.
        '''
         
        min_weight = self.weight_initialization_parameters['random_min_weight']
        max_weight = self.weight_initialization_parameters['random_max_weight']
         
        self.weights = self.random_generator.rand(self.number_of_synapses) * (max_weight-min_weight) + min_weight
        return
     
    def _generate_fixed_weights_(self):
        '''
        Initialize all the weights to the specified values.
        '''
         
        initial_weight = self.weight_initialization_parameters['initial_weight']
        self.weights = numpy.array([initial_weight] * self.number_of_synapses)
        return
    
    def save_layer(self, root, time):
        '''
        This function stores the connectivity data and the synaptic weights in the hdf5 group passed as an argument.
        '''
        import h5py
        
        # Store the hdf5 group object for future links
        self.hdf5_group = root
        
        # Define the attributes of the layer
        root.attrs['name'] = self.__name__
        root.attrs['source_layer'] = self.source_layer.__name__
        root.attrs['target_layer'] = self.target_layer.__name__
        
        # Store the source and target indexes and the weights of the layer
        n_connections = self.source_index.shape[0]
        self.network_record['connections_dset'] = root.create_dataset('connections', data = numpy.array([self.source_index,self.target_index]))
        self.network_record['weights_dset'] = root.create_dataset('weights', (n_connections+1,1), data = numpy.append([time],self.weights).astype(numpy.float32),maxshape=(n_connections+1,None))
        
        return
    
    def add_weights_to_record(self, time):
        '''
        This function stores the connectivity data and the synaptic weights in the hdf5 group passed as an argument.
        '''
        if (self.weight_recording):
            weight_row = numpy.append([time],self.weights).astype(numpy.float32)
            dset_shape = self.network_record['weights_dset'].shape
            self.network_record['weights_dset'].resize((dset_shape[0],dset_shape[1]+1))
            self.network_record['weights_dset'][:,-1] = weight_row
            
        return
    
    def load_layer(self, root):
        '''
        This function loads the connectivity data and the synaptic weights in the hdf5 group passed as an argument.
        '''
        import h5py
        
        # Store the hdf5 group object for future links
        self.hdf5_group = root
        root.associated_object = self
        
        # Load the attributes of the layer
        #self.__name__ = root.attrs['name']
        
        # Load the source and target indexes and the weights of the layer
        target_index = root['connections'][1,:]
        source_index = root['connections'][0,:]
        
        # Select only those synapses targetting local neurons
        selected_target = numpy.where(self.target_layer.is_local_node[target_index]) [0]
        self.target_index = target_index[selected_target]
        self.source_index = source_index[selected_target]
        self.weights = root['weights'][selected_target+1,-1]
        
        # Set the number of synapses
        self.number_of_synapses = self.target_index.shape[0]
        
        return

def _calculate_distance_(source_coord, target_coord):
    '''
    Calculate the euclidean distance between the source and target coordinates
    '''
    return numpy.linalg.norm(numpy.array(source_coord)-numpy.array(target_coord))

def _get_target_cell_indexes_(indexes, interm_to_target_syn_layer):
    '''
    Extract the indexes of the target cells of those cells indicated by indexes in the interm_layer.
    '''
    return numpy.array([interm_to_target_syn_layer.target_index[interm_to_target_syn_layer.source_index==index] for index in indexes])
    
def _get_source_cell_indexes_(indexes, interm_to_target_syn_layer):
    '''
    Extract the indexes in the interm_layer of the sources cells of those cells indicated by indexes in the target_layer.
    '''
    return numpy.array([interm_to_target_syn_layer.source_index[interm_to_target_syn_layer.target_index==index] for index in indexes])
    