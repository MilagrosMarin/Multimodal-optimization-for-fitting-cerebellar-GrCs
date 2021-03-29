
import logging
import Feature
import numpy

# Get logger with default level to INFO
logger = logging.getLogger('FeatureAnalysis')

class Latency (Feature.Feature):

    def __init__(self,name, simulation, config_dict):
        super(Latency, self).__init__(name,simulation, config_dict)
        return


    def calculate(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''

        init_time = self.stimulation_dict['init']
        end_time = self.stimulation_dict['end']

        gtime,gcell_id = self.data_provider.get_spike_activity(neuron_layer = self.layer, init_time = init_time, end_time = end_time)

        num_neurons = self.data_provider.get_number_of_elements(layer=self.layer)

        # Initialize an array with all the elements to the stimulation length
        latency = numpy.repeat(end_time-init_time,num_neurons)

        for sp_time, nid in zip(gtime,gcell_id):
            if latency[nid]>(sp_time-init_time):
                latency[nid] = sp_time-init_time

        av_latency = numpy.average(latency)

        logger.debug('Average latency in layer %s: %s', self.layer, str(av_latency))

        score = abs(av_latency-self.target_value)*self.weight

        logger.debug('Calculated score %s in layer %s', str(score),self.layer)

        return score

    def calculate_feature(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''

        init_time = self.stimulation_dict['init']
        end_time = self.stimulation_dict['end']

        gtime,gcell_id = self.data_provider.get_spike_activity(neuron_layer = self.layer, init_time = init_time, end_time = end_time)

        num_neurons = self.data_provider.get_number_of_elements(layer=self.layer)

        # Initialize an array with all the elements to the stimulation length
        latency = numpy.repeat(end_time-init_time,num_neurons)

        for sp_time, nid in zip(gtime,gcell_id):
            if latency[nid]>(sp_time-init_time):
                latency[nid] = sp_time-init_time

        av_latency = numpy.average(latency)

        logger.debug('Average latency in layer %s: %s', self.layer, str(av_latency))

        return latency[0]

    def max_value(self):
        init_time = self.stimulation_dict['init']
        end_time = self.stimulation_dict['end']

        av_latency = end_time-init_time

        logger.debug('Using default first spike latency value in layer %s: %s', self.layer, av_latency)

        score = abs(av_latency - self.target_value) * self.weight

        logger.debug('Calculated score %s in layer %s', str(score), self.layer)

        return score
