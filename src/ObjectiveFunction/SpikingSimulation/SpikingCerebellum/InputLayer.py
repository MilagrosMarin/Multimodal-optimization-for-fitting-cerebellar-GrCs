'''
Created on March 22, 2014

@author: Jesus Garrido (jgarridoalcazar at gmail.com)
'''
import logging
import numpy
import time

logger = logging.getLogger('Simulation')

class InputLayer(object):
    '''
    This class defines an input layer to a neural network and the data needed to generate it in a simulator.
    '''
    
    def __init__(self,**kwargs):
        '''
        Constructor of the class. It creates a new input layer (composed of fibers). Either the number of neurons or the density of neurons and
        the dimensions (length, width, height)-tuple has to be specified 
        @param number_of_neurons: Number of cells in the layer (optional).
        @param register_activity: Boolean parameter indicating whether the activity in this layer will be registered.
        @param minindex: Index of the first cell in the network (optional).
        @param size: (length, width, height)-tuple with the size of the volume to fill (in mm) (optional).
        @param density_of_neurons: Density of neurons in the volume (optional).
        @param random_generator: The random number generator to use (optional).
        @param soma_size: Size of the soma for this neuron layer (optional). It should be used only for visualization purposes.
        @param load_from_file: h5py group from which the neuron layer positions have to be loaded (optional).
        @param load_state_vars: List of neuron state variables to load from the file
        '''
        
        # Read name
        self.__name__ = kwargs.pop('name')
        
        self.number_of_neurons = None
        self.density_of_neurons = None
        self.size = None
        
        # Read number of neurons
        if ('number_of_neurons' in kwargs):
            self.number_of_neurons = kwargs.pop('number_of_neurons')
        else:
            logger.warning('Non-specified number of neurons in layer %s',self.__name__)
            
        # Read volume size
        if ('size' in kwargs):
            self.size = numpy.array(kwargs.pop('size'))
            # Calculate the total volume of the dice
            self.volume = numpy.prod(self.size)
        else:
            logger.warning('Non-specified volume size in layer %s',self.__name__)
            
        # Read volume size
        if ('density_of_neurons' in kwargs):
            self.density_of_neurons = kwargs.pop('density_of_neurons')
        else:
            logger.warning('Non-specified density of neurons in layer %s',self.__name__)
            
        if self.number_of_neurons is not None:
            if self.size is not None:
                if self.density_of_neurons is not None:
                    logger.warning('Both number of neurons and density of neurons specified in layer %s. Density of neurons will be discarded',self.__name__)
                self.density_of_neurons = self.number_of_neurons / self.volume
        elif self.size is not None and self.density_of_neurons is not None: 
            self.number_of_neurons = int(round(self.volume * self.density_of_neurons))
        else:
            logger.error('Non-specified neither number of neurons nor (density of neurons and size) in layer %s',self.__name__)
            raise Exception('Non-DefinedProperty')
                
        
        # Read register activity parameter
        if ('register_activity' in kwargs):
            self.register_activity = kwargs.pop('register_activity') 
        else:
            logger.warning('Non-specified register_actiity parameter in layer %s. Using default value false',self.__name__)
            self.register_activity = False
            
        # Read minindex parameter
        if ('minindex' in kwargs):
            self.MinIndex = kwargs.pop('minindex') 
        else:
            self.MinIndex = None
            
        # Read seed parameter
        if ('random_generator' in kwargs):
            self.random_generator = kwargs.pop('random_generator') 
        else:
            self.random_generator = numpy.random.RandomState(int(time.time()))
            
        if ('soma_size' in kwargs):
            self.soma_size = kwargs.pop('soma_size')
        else:
            self.soma_size = None
            
        if ('load_from_file' in kwargs):
            self.load_from_file = kwargs.pop('load_from_file')
        else:
            self.load_from_file = None
            
        if ('load_state_vars' in kwargs):
            self.load_state_vars = kwargs.pop('load_state_vars')
        else:
            self.load_state_vars = None
        
        if ('save_state_vars' in kwargs):
            self.save_state_vars = kwargs.pop('save_state_vars')
        else:
            self.save_state_vars = None
            
        self.neuron_states = dict()
            
        # Check whether additional parameters have been used.
        for param in kwargs:
            logger.warning('Unrecognized parameter %s in layer %s',param,self.__name__)
            
        # Locate the elements if required
        if self.size is not None:
            self._locate_elements_()
            
        return
            
            
    def _locate_elements_(self):
        '''
        This function locate every neuron in its specific place. The size needs to be specified before calling this function
        '''
        
        if self.size is None:
            logger.error('Non-specified size in layer %s. Neurons cannot be spatially located',self.__name__)
            raise Exception('Non-DefinedProperty')
                
        n_dimensions = len(self.size)
        
        if (self.load_from_file is None):
            self.relative_positions = self.random_generator.uniform(0,1,(self.number_of_neurons, n_dimensions)).astype(numpy.float32)
        else:
            self.load_layer(self.load_from_file)
        
        self._share_positions_() 
        
        self.is_local_node = numpy.empty((self.number_of_neurons),dtype=numpy.bool)
        self.is_local_node[:] = True
        return
    
    def _share_positions_(self):
        '''
        This function sends the positions stored in the first process to all the MPI processes. This function will be reimplemented in
        inheriting classes. 
        '''
        return
    
    def get_relative_coordinates(self):
        return self.relative_positions
    
    def get_absolute_coordinates(self):
        return self.relative_positions * self.size
    
    def save_layer(self, root, time):
        '''
        This function stores the relative positions of the neurons and other attributes in the hdf5 group passed as an argument.
        '''
        import h5py
        
        # Store the hdf5 group object for future links
        self.hdf5_group = root
        
        # Define the attributes of the layer
        root.attrs['name'] = self.__name__
        root.attrs['size'] = self.size
        
        # Define the relative positions of the layer
        positions_dataset = root.create_dataset('relative_positions', data = self.relative_positions)
        
        if self.neuron_states:
            self.dataset = dict()
            
            for key, value in self.neuron_states.iteritems():
                n_neurons = value.shape[0]
                self.dataset[key] = root.create_dataset(key, (n_neurons+1,1), data = numpy.append([time],value).astype(numpy.float32),maxshape=(n_neurons+1,None))
        
        
        return
    
    def add_state_to_record(self, time):
        '''
        This function stores the relative positions of the neurons and other attributes in the hdf5 group passed as an argument.
        '''
        if self.neuron_states:
            for key, value in self.neuron_states.iteritems():
                state_row = numpy.append([time],value).astype(numpy.float32)
                dset_shape = self.dataset[key].shape
                self.dataset[key].resize((dset_shape[0],dset_shape[1]+1))
                self.dataset[key][:,-1] = state_row
                        
        return
    
    def load_layer(self, root):
        '''
        This function load the relative positions of the neurons and other attributes in the hdf5 group passed as an argument.
        '''
        import h5py
        
        # Store the hdf5 group object for future links
        self.hdf5_group = root
        root.associated_object = self
        
        # Load the attributes of the layer
        self.__name__ = root.attrs['name']
        self.size = root.attrs['size']
        
        # Load the relative positions of the layer
        self.relative_positions = root['relative_positions'][:,:]
        
        # Set the number of neurons
        self.number_of_neurons = self.relative_positions.shape[0]
        
        # Get NEST model recordable variables
        if (self.load_state_vars):
                
            self.neuron_states = dict()
                
            # If only an string has been used, embed it in an array
            if isinstance(self.load_state_vars,str):
                self.load_state_vars = [self.load_state_vars]
                    
            # Get recording variables in this cell model
            for var in self.load_state_vars:
                if var in root:
                    self.neuron_states[var] = root[var][-1,1:]
                else:
                    logger.warning('%s state variable does not exist in h5 file. Ignoring',var)
        
        return
                
