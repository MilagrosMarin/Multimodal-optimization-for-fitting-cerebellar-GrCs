
import logging
import Feature
import numpy

# Get logger with default level to INFO
logger = logging.getLogger('FeatureAnalysis')

class BurstFrequency (Feature.Feature):

    def __init__(self,name, simulation, config_dict):
        super(BurstFrequency, self).__init__(name,simulation, config_dict)
        return


    def calculate(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''

        osc_frequency = self.stimulation_dict['frequency']

        init_cycle = self.feat_config_dict['init_cycle']
        end_cycle = self.feat_config_dict['end_cycle']

        gtime,gcell_id = self.data_provider.get_spike_activity(neuron_layer = self.layer)
        #logger.debug('total_peaktime (%s) : %s',len(gtime),gtime)

        #vm_time, vm_cell_id, vm = self.data_provider.get_state_variable(state_variable='Vm', neuron_layer = self.layer)
        #  i_events = nest.GetStatus(self.data_provider.nest_layer.nest_multimeter,'events')[0]
        #logger.debug('vm time: %s, current_vm: %s', vm_time[:10], vm[:10])

        gcycle = (gtime * osc_frequency).astype(int) + 1
        # init cycle starts at 0.; end cycle not included!
        #gcycle = (gtime * osc_frequency).astype(int)

        num_neurons = self.data_provider.get_number_of_elements(layer=self.layer)

        burst_freq_per_neuron = numpy.zeros((num_neurons))

        for neuid in range(num_neurons):
            sel_spikes = gcell_id==neuid

            #freq=0.
            freq = []
            for cycle in range(init_cycle,end_cycle): #Starts in 1; end_cycle not included
                sel_spikes_cycle = numpy.logical_and(sel_spikes,gcycle==cycle)
                sp_times = gtime[sel_spikes_cycle]
                #logger.debug('cycle:%s',cycle)

                if (len(sp_times)==0):
                    # If no spike is fired, assume frequency is stimulation frequency
                    #freq += osc_frequency
                    freq.append(0.)

                elif (len(sp_times)==1):
                    # If only one spike is fired, assume frequency is stimulation frequency
                    #freq += osc_frequency
                    freq.append(0.)

                elif (len(sp_times)>1):
                    # If 2+ spikes fired, calculate average ISI
                    ISI = numpy.diff(sp_times)
                    av_ISI = numpy.average(ISI)
                    # freq += (1./av_ISI)
                    freq.append(1./av_ISI)

                #logger.debug('spike times (%s) in cycle %s: %s', len(sp_times),str(cycle), str(sp_times))
                #logger.debug('ISI: %s, av_ISI: %s, freq: %s',str(ISI), str(av_ISI), str(freq))

            #burst_freq_per_neuron[neuid] = freq/(end_cycle-init_cycle)
            burst_freq_per_neuron[neuid] = numpy.average(freq)

        layer_average = numpy.average(burst_freq_per_neuron)

        logger.debug('Obtained average burst frequencies in layer %s: Average per neuron: %s Mean burst frequency: %s', self.layer, str(burst_freq_per_neuron), str(layer_average))

        score = abs(layer_average-self.target_value)*self.weight

        logger.debug('Calculated score %s in layer %s', str(score),self.layer)

        logger.error('%s= %s #(# cycles %s with Mean Freq = %s)', self.name, freq, (end_cycle-init_cycle),burst_freq_per_neuron[0])

        return score

    def calculate_feature(self):
        '''
        This method calculates the value of the feature/score with the current simulation
        '''

        osc_frequency = self.stimulation_dict['frequency']

        init_cycle = self.feat_config_dict['init_cycle']
        end_cycle = self.feat_config_dict['end_cycle']

        gtime,gcell_id = self.data_provider.get_spike_activity(neuron_layer = self.layer)
        #logger.debug('total_peaktime (%s) : %s',len(gtime),gtime)

        #vm_time, vm_cell_id, vm = self.data_provider.get_state_variable(state_variable='Vm', neuron_layer = self.layer)
        #  i_events = nest.GetStatus(self.data_provider.nest_layer.nest_multimeter,'events')[0]
        #logger.debug('vm time: %s, current_vm: %s', vm_time[:10], vm[:10])

        gcycle = (gtime * osc_frequency).astype(int) + 1

        num_neurons = self.data_provider.get_number_of_elements(layer=self.layer)

        burst_freq_per_neuron = numpy.zeros((num_neurons))

        for neuid in range(num_neurons):
            sel_spikes = gcell_id == neuid

            # freq=0.
            freq = []
            for cycle in range(init_cycle, end_cycle):  # Starts in 1; end_cycle not included
                sel_spikes_cycle = numpy.logical_and(sel_spikes, gcycle == cycle)
                sp_times = gtime[sel_spikes_cycle]
                # logger.debug('cycle:%s',cycle)

                if (len(sp_times)==0):
                    # If no spike is fired, assume frequency is stimulation frequency
                    #freq += osc_frequency
                    freq.append(0.)

                if (len(sp_times) == 1):
                    # If only one spike is fired, assume frequency is stimulation frequency
                    # freq += osc_frequency
                    freq.append(0.)

                elif (len(sp_times) > 1):
                    # If 2+ spikes fired, calculate average ISI
                    ISI = numpy.diff(sp_times)
                    av_ISI = numpy.average(ISI)
                    # freq += (1./av_ISI)
                    freq.append(1. / av_ISI)

                    # logger.debug('spike times (%s) in cycle %s: %s', len(sp_times),str(cycle), str(sp_times))
                    # logger.debug('ISI: %s, av_ISI: %s, freq: %s',str(ISI), str(av_ISI), str(freq))

            # burst_freq_per_neuron[neuid] = freq/(end_cycle-init_cycle)

            burst_freq_per_neuron[neuid] = numpy.average(freq)

        layer_average = numpy.average(burst_freq_per_neuron)

        logger.debug('Obtained average burst frequencies in layer %s: Average per neuron: %s Mean burst frequency: %s', self.layer, str(burst_freq_per_neuron), str(layer_average))

        return layer_average

    def max_value(self):
        mean_freq = 0

        logger.debug('Using default burst frequency in layer %s: %s', self.layer, mean_freq)

        score = abs(mean_freq - self.target_value) * self.weight

        logger.debug('Calculated score %s in layer %s', str(score), self.layer)

        return score