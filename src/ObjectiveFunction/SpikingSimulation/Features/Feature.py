import logging
from SpikingSimulation.Utils.Logger import InitializeLogger, Logger2File

InitializeLogger('FeatureAnalysis')

# Get logger with default level to INFO
logger = logging.getLogger('FeatureAnalysis')

import abc

class Feature(object):
    '''
    This class implements a generic feature for EA optimization.
    '''

    def __init__(self, name, simulation, config_dict):
        self.name = name
        self.feat_config_dict = config_dict
        self.simulation = simulation
        self.data_provider = simulation.cerebellum
        return

    def initialize(self):

        if 'verbosity' not in self.simulation.config_options['simulation']:
            self.simulation.config_options['simulation']['verbosity'] = 'debug'
            logger.warning('Non-specified simulation verbosity. Using default value %s',
                           self.simulation.config_options['simulation']['verbosity'])

        numeric_level = getattr(logging, self.simulation.config_options['simulation']['verbosity'].upper(), None)
        if not isinstance(numeric_level, int):
            self.simulation.config_options['simulation']['verbosity'] = 'info'
            numeric_level = getattr(logging, self.simulation.config_options['simulation']['verbosity'].upper(), None)
            logger.warning('Invalid simulation verbosity. Using default value %s',
                           self.simulation.config_options['simulation']['verbosity'])
            raise ValueError('Invalid log level: %s' % self.simulation.config_options['simulation']['verbosity'])

        logger.setLevel(numeric_level)

        if not 'layer' in self.feat_config_dict:
            logger.error('Layer name has not been specified in the feature section %s', self.name)
            raise Exception('NonSpecifiedLayer')
        else:
            self.layer = self.feat_config_dict['layer']

        if self.layer not in self.data_provider.layer_map:
            logger.error('Invalid cell layer %s in feature analysis %s', self.layer, self.name)
            raise Exception('InvalidParameter', 'layer')

        if not 'stimulation' in self.feat_config_dict:
            logger.error('Stimulation name has not been specified in the feature section %s', self.name)
            raise Exception('NonSpecifiedStimulation')

        if self.feat_config_dict['stimulation'] not in self.simulation.config_options:
            logger.error('Invalid stimulation name %s in feature analysis %s', self.feat_config_dict['stimulation'], self.name)
            raise Exception('InvalidParameter', 'layer')
        else:
            self.stimulation_dict = self.simulation.config_options[self.feat_config_dict['stimulation']]

        if not 'target_value' in self.feat_config_dict:
            logger.warning('Target value has not been specified in the feature section %s. Using default value 0', self.name)
            self.target_value = 0.0
        else:
            self.target_value = self.feat_config_dict['target_value']

        if not 'weight' in self.feat_config_dict:
            logger.warning('Weight value has not been specified in the feature section %s. Using default value 1.0', self.name)
            self.weight = 1.0
        else:
            self.weight = self.feat_config_dict['weight']

            logger.info('Analyzing feature in section %s', self.name)



        return


    @abc.abstractmethod
    def calculate(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''
        return

    @abc.abstractmethod
    def calculate_feature(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''
        return

    @abc.abstractmethod
    def max_value(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''
        return

