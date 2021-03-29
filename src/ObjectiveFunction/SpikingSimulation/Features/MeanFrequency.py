
import logging
import Feature
import numpy

# Get logger with default level to INFO
logger = logging.getLogger('FeatureAnalysis')

class MeanFrequency (Feature.Feature):

    def __init__(self,name, simulation, config_dict):
        super(MeanFrequency, self).__init__(name,simulation, config_dict)
        return


    def calculate(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''

        init_time = self.stimulation_dict['init']
        end_time = self.stimulation_dict['end']

        gtime,gcell_id = self.data_provider.get_spike_activity(neuron_layer = self.layer, init_time = init_time, end_time = end_time)

        num_spikes = gtime.size
        num_neurons = self.data_provider.get_number_of_elements(layer=self.layer)
        mean_freq = num_spikes / (float(num_neurons) * (end_time-init_time))

        logger.debug('Obtained %s spikes with %s neurons in layer %s: Mean frequency: %s', str(num_spikes), str(num_neurons), self.layer, str(mean_freq))

        score = abs(mean_freq-self.target_value)*self.weight

        logger.debug('Calculated score %s in layer %s', str(score),self.layer)

        return score

    def calculate_feature(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''

        init_time = self.stimulation_dict['init']
        end_time = self.stimulation_dict['end']

        gtime,gcell_id = self.data_provider.get_spike_activity(neuron_layer = self.layer, init_time = init_time, end_time = end_time)

        num_spikes = gtime.size
        num_neurons = self.data_provider.get_number_of_elements(layer=self.layer)
        mean_freq = num_spikes / (float(num_neurons) * (end_time-init_time))

        logger.debug('Obtained %s spikes with %s neurons in layer %s: Mean frequency: %s', str(num_spikes), str(num_neurons), self.layer, str(mean_freq))

        return mean_freq

    def max_value(self):
        init_time = self.stimulation_dict['init']
        end_time = self.stimulation_dict['end']

        if self.target_value != 0.:
            mean_freq = 0.

            logger.debug('Using default mean frequency in layer %s: %s', self.layer, mean_freq)

            score = abs(mean_freq - self.target_value) * self.weight

            logger.debug('Calculated score %s in layer %s', str(score), self.layer)

        else:
            mean_freq = 250.

            logger.debug('Using default mean frequency in layer %s: %s', self.layer, mean_freq)

            score = abs(mean_freq - self.target_value) * self.weight

            logger.debug('Calculated score %s in layer %s', str(score), self.layer)

        return score