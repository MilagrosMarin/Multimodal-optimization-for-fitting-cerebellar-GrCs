'''
Created on April 22, 2014

@author: Jesus Garrido (jgarridoalcazar at gmail.com)
'''

import InputLayer
import logging

logger = logging.getLogger('Simulation')

class NeuronLayer(InputLayer.InputLayer):
    '''
    This class defines a neuron layer and the data needed to generate it in a simulator.
    '''
    
    template_model_parameters = {
                         'ConductanceLIF': ['cm','texc','tinh','grest','eexc','einh','erest','eth','tref', 'tau_minus', 'tau_istdp'],
                         'ConductanceLIFSym': ['cm','texc','tinh','grest','eexc','einh','erest','eth','tref', 'tau_minus', 'tau_istdp','tau_sym'],
                         'ConductanceLIFwIP': ['cm','texc','tinh','grest','eexc','einh','erest','eth','tref', 'tau_ip','beta_ip','epsilon_rc_ip','epsilon_rr_ip','tau_minus', 'tau_istdp', 'max_cm'],
                         'ConductanceLIFwIPSym': ['cm','texc','tinh','grest','eexc','einh','erest','eth','tref', 'tau_ip','beta_ip','epsilon_rc_ip','epsilon_rr_ip','tau_istdp','tau_minus', 'max_cm'],
                         'ConductanceLIFwAT': ['cm','texc','tinh','grest','eexc','einh','erest','eth','tref', 'tau_th','th_cons','tau_minus', 'tau_istdp'],
                         'ConductanceLIFwATSym': ['cm','texc','tinh','grest','eexc','einh','erest','eth','tref', 'tau_th','th_cons','tau_istdp','tau_minus'],
                         'ConductanceLIFStowIP': ['cm','texc','tinh','grest','eexc','einh','erest','eth','tref', 'tref_abs', 'tau_sym', 'tau_istdp','ip_rate','target_freq','tau_minus'],
                         'CurrentLIF': ['cm','grest','erest','eth','tref','tau_minus', 'tau_istdp'],
                         'ConductanceExpAdEx': ['cm','texc','tinh','grest','eexc','einh','erest','eth','espike','tw','delta_t','a','b','tref','tau_minus','vreset']
                        }
    def __init__(self,**kwargs):
        '''
        Constructor of the class. It creates a new neuron layer.
        @param number_of_neurons: Number of cells in the layer.
        @param cell_model: Cell model name. For the moment, only 'ConductanceLIF' is recognized.
        @param cell_model_parameters: Cell model parameters. For 'ConductanceLIF' the following
        parameters should be included: 'cm', 'texc', 'tinh', 'grest', 'eexc', 'einh', 'erest', 'eth', 'tref'.
        @param register_activity: Boolean parameter indicating whether the activity in this layer will be registered.
        @param is_output: Boolean parameter indicating wheter the activity in this layer will be sent to an external subsystem.
        @param record_vars: List with the name of the states variables to be recorded in this layer (e.g., Vm, Gexc, Ginh)
        @param record_step: Time-step for recorded state vars (in s)
        '''
        
        # Read cell model and its properties
        if ('cell_model' in kwargs):
            self.cell_model = kwargs.pop('cell_model')
            if (self.cell_model in self.template_model_parameters):
                self.cell_model_parameters = {}
                for param in self.template_model_parameters[self.cell_model]:
                    if (param in kwargs):
                        self.cell_model_parameters[param] = kwargs.pop(param)
                    else:
                        logger.warning('Non-specified cell model parameter: %s in layer %s. Using default value', param, kwargs['name'])
            else:
                logger.error('Unknown cell model: %s', self.cell_model)
                raise Exception('UnknownCellModel')                         
        else:
            logger.error('Non-specified cell model')
            raise Exception('Non-DefinedProperty')
        
        if ('record_vars' in kwargs):
            self.record_vars = kwargs.pop('record_vars')
        else:
            self.record_vars = None
            
        if ('record_step' in kwargs):
            self.record_step = kwargs.pop('record_step')
        else:
            self.record_step = 1e-3            
        
        super(NeuronLayer, self).__init__(**kwargs)
        
    def equal_cell_model(self, neuronLayer1):
        '''
        It compares the cell model of this neuron layer with the one in the neuron layer passed as a parameter.
        @param neuronLayer1 The neuron whose cell model needs to be compared with.
        @return True if the cell models of the two neuron layers represent the same one (same name and parameters). False otherwise.
        '''

        # Checking the type of the parameter
        if (not(isinstance(neuronLayer1, NeuronLayer))):
            return False
        
        # Checking the number of the cell model name
        if (self.cell_model != neuronLayer1.cell_model):
            return False

        # Checking parameter values
        return (self.cell_model_parameters == neuronLayer1.cell_model_parameters)