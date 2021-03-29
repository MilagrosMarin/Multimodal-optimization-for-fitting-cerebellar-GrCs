
"""
This is an adaptation of the code generated for the definition, analysis and evaluation of a suitable fitness function for calculating the most representative spiking features of cerebellar granule cells published in M. Marin et al. 2020.

"""

import pprint
pp = pprint.PrettyPrinter(indent=4)
#%matplotlib notebook
#import seaborn as sns
#sns.set(style="darkgrid")

import matplotlib.pyplot as plt
import nest
import numpy
import pdb
import sys 
import os
from scipy import stats


sys.path.insert(1,'./src')
import SpikingSimulation.CurrentSimulation as CurrentSimulation
import SpikingSimulation.Features.MeanFrequency as MeanFrequency
import SpikingSimulation.Features.Latency as Latency
import SpikingSimulation.Features.BF_sd as BF_sd

def new_initialisation(config_file,simulation_time,first_params):
    nest.ResetKernel()
    simulation = CurrentSimulation.CurrentSimulation(config_file=config_file)
    simulation.config_options['simulation']['verbosity'] = 'Error'
    simulation.config_options['simulation']['seed'] = 12345
    simulation.config_options['simulation']['time'] = simulation_time
    simulation.config_options['mflayer']['cm'] = first_params['cm']
    simulation.config_options['mflayer']['grest'] = first_params['grest']
    simulation.config_options['mflayer']['erest'] = first_params['erest']
    simulation.config_options['mflayer']['eth'] = first_params['eth']
    simulation.config_options['mflayer']['espike'] = first_params['espike']
    simulation.config_options['mflayer']['a'] = first_params['a']
    simulation.config_options['mflayer']['b'] =  first_params['b']
    simulation.config_options['mflayer']['tw'] = first_params['tw']
    simulation.config_options['mflayer']['delta_t'] =  first_params['delta_t']
    simulation.config_options['mflayer']['tref'] = first_params['tref']
    simulation.config_options['mflayer']['vreset'] = first_params['vreset']
    simulation.config_options['mflayer']['record_vars'] = ['Vm','w']
    simulation.config_options['mflayer']['record_step'] = 1.0e-4
    return simulation

def create_dict_case(dict_name):
    dict_name = dict()
    dict_name['scores']=dict() 
    dict_name['plot']=dict()
    dict_name['plot']['mean_frequency']=dict()
    dict_name['plot']['latency']=dict()
    dict_name['plot']['burst_frequency']=dict()
    dict_name['plot']['burst_frequency']=dict()
    return dict_name

def score_calculation(dict_name, config_file, simulation_time):
    """
    This function calculates the feature scores from simulated neurons in order to evaluate the features of spiking resonance,
    repetitive spike discharge and latency to the first spikes of cerebellar granule cells obtained after the execution of the
    multimodal algorithm in M.Marin et al. (2021).
    
    The score of burst frequency is based on the experimental data from DAngelo et al. 2001, where the stimulation protocol 
    consists of the injection of a sinusoidal current of 6 pA-amplitude and 8 pA-amplitude plus a offset current of 12 pA. 
    
    The feature score of spiking resonance is based on the summated distance between simulated values and biological values
    for the burst frequencies, as presented in M. Marin et al. (2020). In this publication, we differenciated between two
    ways of calculation of the total score: equation (3) refers to the distance between simulated and real values of features;
    equation (4) is a specific formula for the calculation of stable burst frequencies, including the standard deviation in
    the calculus. For this reason, here we calculate the total score based on both premises: the variable defined as
    "Burst_Frequency" refers to the distance between real and simulated values; the variable named "bf_SD" refers to the
    feature score based on the equation (4) where the standard deviation is used as a penalizer. 
    
    The features of repetitive spike discharge and latency to the first spike are based on the experimental data from Masoli 
    et al. 2017. The feature score is calculated as explained in M. Marin et al. (2020), the summated difference between the
    real and the simulated values. 
    
    The calculations are saved in the dictionary created before (named dict_name). 
    
    """
    
    #### SIMULATION AND CALCULATION OF BURST FREQUENCIES  ####
    
    #------------------------------------------------------------
    # 1. During a 6 pA-amplitude sinusoidal current injection
    #------------------------------------------------------------
    # Stimulation protocol
    sine_current_amplitude =[6.0e-12] # (A)
    sine_current_phase =270. # (degrees)
    sine_current_offset = 12.0e-12 # (A)

    sinu_freq_6Hz = [0.58, 2.12, 4.04, 5.96, 8.08, 10.19] #Hz
    target_values_6Hz = [41.43, 49.29, 54.00, 59.29, 55.00, 45.71] #Hz

    
    init_cycles_6Hz_whole = [2,5,9,12,17,21]
    end_cycles_6Hz_whole = [12,15,19,22,27,31]
    
    #------------------------------------------------------------   
    error_total_6 =0.
    error_total_8 =0.
    score_total =0.
    score2_total =0.
    
    for c,freq in enumerate(sinu_freq_6Hz): 
        simulation = new_initialisation(config_file,simulation_time,dict_name['param_configuration'])
        simulation.config_options['sin_current{}'.format(c)] = dict()
        simulation.config_options['sin_current{}'.format(c)]['amplitude'] = sine_current_amplitude[0]
        simulation.config_options['sin_current{}'.format(c)]['phase'] = sine_current_phase
        simulation.config_options['sin_current{}'.format(c)]['offset'] = sine_current_offset
        simulation.config_options['sin_current{}'.format(c)]['frequency'] = freq

        simulation.initialize()
        #print nest.GetStatus(simulation.cerebellum.ac_generator[0].tolist())    
        simulation.run_simulation()

        #BURST FREQUENCY FEATURE
        feature_dict = dict()
        feature_dict['layer'] = 'mflayer'
        feature_dict['stimulation'] = 'sin_current{}'.format(c)
        feature_dict['init_cycle'] = init_cycles_6Hz_whole[c]
        feature_dict['end_cycle'] = end_cycles_6Hz_whole[c]
        feature_dict['target_value'] = target_values_6Hz[c]
        feature_dict['weight'] = 1.0

        # CALCULATION OF BURST FREQUENCY ACCORDING TO EQUATION (4) (CONSIDERING THE STANDARD DEVIATION) 
        feat_obj = BF_sd.BF_sd('feat_key',simulation,feature_dict)
        feat_obj.initialize()
        layer_average = feat_obj.calculate_feature()[0]
        error = feat_obj.calculate_feature()[1]
        score = feat_obj.calculate()          
        error_total_6 += error
        score_total += score #Eq. 4
      
    #------------------------------------------------------------
    # 2. During an 8 pA-amplitude sinusoidal current injection
    #------------------------------------------------------------
    # Stimulation protocol
    sine_current_amplitude = [8.0e-12] # (A)
    sine_current_phase =270. # degrees
    sine_current_offset = 12.0e-12 # (A)

    sinu_freq_8Hz = [0.58, 2.12, 4.04, 5.96, 8.08, 10.19,12.31,14.23] #Hz
    target_values_8Hz = [45.00, 55.71, 60.00, 65.71, 66.43, 64.29,58.57,50.00] #Hz

    init_cycles_8Hz_whole = [2,5,9,12,17,21,25,29]
    end_cycles_8Hz_whole = [12,15,19,22,27,31,35,39]
    
    #------------------------------------------------------------

    for c,freq in enumerate(sinu_freq_8Hz): 
        simulation = new_initialisation(config_file,simulation_time,dict_name['param_configuration'])
        simulation.config_options['sin_current{}'.format(c)] = dict()
        simulation.config_options['sin_current{}'.format(c)]['amplitude'] = sine_current_amplitude[0]
        simulation.config_options['sin_current{}'.format(c)]['phase'] = sine_current_phase
        simulation.config_options['sin_current{}'.format(c)]['offset'] = sine_current_offset
        simulation.config_options['sin_current{}'.format(c)]['frequency'] = freq

        simulation.initialize()
        #print nest.GetStatus(simulation.cerebellum.ac_generator[0].tolist())    
        simulation.run_simulation()

        #BURST FREQUENCY FEATURE
        feature_dict = dict()
        feature_dict['layer'] = 'mflayer'
        feature_dict['stimulation'] = 'sin_current{}'.format(c)
        feature_dict['init_cycle'] = init_cycles_8Hz_whole[c]
        feature_dict['end_cycle'] = end_cycles_8Hz_whole[c]
        feature_dict['target_value'] = target_values_8Hz[c]
        feature_dict['weight'] = 1.0

        # CALCULATION OF BURST FREQUENCY ACCORDING TO EQUATION (4) (CONSIDERING THE STANDARD DEVIATION) 
        feat_obj = BF_sd.BF_sd('feat_key',simulation,feature_dict)
        feat_obj.initialize()
        layer_average = feat_obj.calculate_feature()[0]
        error = feat_obj.calculate_feature()[1]
        score = feat_obj.calculate()          
        error_total_8 += error
        score_total += score
    #pp.pprint(scores)
    
  # SAVE THE SCORES IN THE NEURON DICTIONARY
#    dict_name['scores']['total_score_SD_6'] = error_total_6
#    dict_name['scores']['total_score_SD_8'] = error_total_8
    dict_name['scores']['feature_Burst_Frequency'] = score_total

    #------------------------------------------------------------

    #### SIMULATION AND CALCULATION OF REPETITIVE FIRE DISCHARGE (I-F Curves)  ####
    
    #------------------------------------------------------------
    # 3. During a step-current injection
    #------------------------------------------------------------
    # Stimulation protocol
    init_time = 0.
    end_time = 1.
    simulation_time = 1.1
    weights={'MF':1.,'LAT':1000.}
    target_values = {
    'stim1':{'mean_frequency': 30.0, 'latency': 31.9e-3},
    'stim2':{'mean_frequency': 45.0, 'latency': 19.0e-3},
    'stim3':{'mean_frequency': 60.0, 'latency': 14.65e-3} }

    mf_total_score = []
    lat_total_score = []
    
    #CALCULATION  
    for c,amp in enumerate([10.,16.,22.]):
        simulation = new_initialisation(config_file,simulation_time,dict_name['param_configuration'])
        simulation.config_options['pulse_current1'] = dict()
        simulation.config_options['pulse_current1']['init'] = init_time
        simulation.config_options['pulse_current1']['end'] = end_time
        simulation.config_options['pulse_current1']['amplitude'] = amp*1.e-12 # (A)
        simulation.initialize()
        simulation.run_simulation()

        #MEAN FREQUENCY FEATURE
        feature_dict = dict()
        feature_dict['layer'] = 'mflayer'
        feature_dict['stimulation'] = 'pulse_current1'
        feature_dict['target_value'] = target_values['stim{}'.format(c+1)]['mean_frequency']
        feature_dict['weight'] = weights['MF']
        feat_obj = MeanFrequency.MeanFrequency('feat_key',simulation,feature_dict)
        feat_obj.initialize()
        mf_value = feat_obj.calculate_feature()
        mf_score = feat_obj.calculate()
        mf_total_score.append(mf_score)
        
        #LATENCY FEATURE
        feature_dict = dict()
        feature_dict['layer'] = 'mflayer'
        feature_dict['stimulation'] = 'pulse_current1'
        feature_dict['target_value'] = target_values['stim{}'.format(c+1)]['latency']
        feature_dict['weight'] = weights['LAT']
        feat_obj = Latency.Latency('feat_key',simulation,feature_dict)
        feat_obj.initialize()
        lat_value = feat_obj.calculate_feature()
        lat_score = feat_obj.calculate()
        lat_total_score.append(lat_score)
        
  # SAVE THE SCORES IN THE NEURON DICTIONARY
    dict_name['scores']['feature_Mean_Frequency'] = sum(mf_total_score)
    dict_name['scores']['feature_Latency'] = sum(lat_total_score)        

    
def score_representation(dict_name, config_file, simulation_time,file_name,show=True,savefig=False):
    
    """
    This function represents the data shown in the figures of the article. 
    """
    
     #### REPRESENTATION OF SPIKING RESONANCE  ####
    #------------------------------------------------------------
    # During sinusoidal current injections 
    #------------------------------------------------------------        
    #Definition of the sinusoidal current injection
    sine_current_frequency=numpy.arange(0.,30.5,0.5) 
    sine_current_amplitude = [6.0e-12] # (A)
    sine_current_phase =270. # degrees
    sine_current_offset = 12.0e-12 # (A)
    
    # Representation of real values of spiking resonance from D'Angelo et al.(2001)
    sinu_freq_6Hz = [0.58, 2.12, 4.04, 5.96, 8.08, 10.19] 
    target_values_6Hz = [41.43, 49.29, 54.00, 59.29, 55.00, 45.71]
    sinu_freq_8Hz = [0.58, 2.12, 4.04, 5.96, 8.08, 10.19,12.31,14.23]
    target_values_8Hz = [45.00, 55.71, 60.00, 65.71, 66.43, 64.29,58.57,50.00]

    # Selection of burst after a period of stabilization [as explained in M.Marin et al. (2020)]
    init_cycles_6Hz_whole = []
    end_cycles_6Hz_whole = []
    for sinu_freq in sine_current_frequency:
        init_cycles_6Hz_whole.append(int(numpy.ceil(sinu_freq*2)))
        end_cycles_6Hz_whole.append(int(numpy.ceil(sinu_freq*2)+10))

    init_cycles_8Hz_whole = []
    end_cycles_8Hz_whole = []
    for sinu_freq in sine_current_frequency:
        init_cycles_8Hz_whole.append(int(numpy.ceil(sinu_freq*2)))
        end_cycles_8Hz_whole.append(int(numpy.ceil(sinu_freq*2)+10))

    av_burst_freq_6 = []
    sinu_freq_6 = []
    error_6 = []
    
    for c,freq in enumerate(sine_current_frequency): 
        simulation = new_initialisation(config_file,simulation_time,dict_name['param_configuration'])
        simulation.config_options['sin_current{}'.format(c)] = dict()
        simulation.config_options['sin_current{}'.format(c)]['amplitude'] = sine_current_amplitude[0]
        simulation.config_options['sin_current{}'.format(c)]['phase'] = sine_current_phase
        simulation.config_options['sin_current{}'.format(c)]['offset'] = sine_current_offset
        simulation.config_options['sin_current{}'.format(c)]['frequency'] = freq

        simulation.initialize()
        #print nest.GetStatus(simulation.cerebellum.ac_generator[0].tolist())    
        simulation.run_simulation()
        
        #BURST FREQUENCY FEATURE
        feature_dict = dict()
        feature_dict['layer'] = 'mflayer'
        feature_dict['stimulation'] = 'sin_current{}'.format(c)
        feature_dict['init_cycle'] = init_cycles_6Hz_whole[c]
        feature_dict['end_cycle'] = end_cycles_6Hz_whole[c]
        #feature_dict['target_value'] = target_values_6Hz[c-1]
        feature_dict['weight'] = 1.0

        feat_obj = BF_sd.BF_sd('feat_key',simulation,feature_dict)
        feat_obj.initialize()
        layer_average = feat_obj.calculate_feature()[0]
        error = feat_obj.calculate_feature()[1]
        #score = feat_obj.calculate()
        #scores['BF6]['{}'.format(c)]=score

        av_burst_freq_6.append(layer_average) 
        error_6.append(error)
        sinu_freq_6.append(freq)

    sine_current_amplitude = [8.0e-12] # (A)
    sine_current_phase =270. # degrees
    sine_current_offset = 12.0e-12 # (A)

    av_burst_freq_8 = []
    sinu_freq_8 = []
    error_8 = []
    for c,freq in enumerate(sine_current_frequency): 
        simulation = new_initialisation(config_file,simulation_time,dict_name['param_configuration'])
        simulation.config_options['sin_current{}'.format(c)] = dict()
        simulation.config_options['sin_current{}'.format(c)]['amplitude'] = sine_current_amplitude[0]
        simulation.config_options['sin_current{}'.format(c)]['phase'] = sine_current_phase
        simulation.config_options['sin_current{}'.format(c)]['offset'] = sine_current_offset
        simulation.config_options['sin_current{}'.format(c)]['frequency'] = freq

        simulation.initialize()
        #print nest.GetStatus(simulation.cerebellum.ac_generator[0].tolist())    
        simulation.run_simulation()

        #BURST FREQUENCY FEATURE
        feature_dict = dict()
        feature_dict['layer'] = 'mflayer'
        feature_dict['stimulation'] = 'sin_current{}'.format(c)
        feature_dict['init_cycle'] = init_cycles_8Hz_whole[c]
        feature_dict['end_cycle'] = end_cycles_8Hz_whole[c]
        #feature_dict['target_value'] = target_values_8Hz[c-1]
        feature_dict['weight'] = 1.0

        feat_obj = BF_sd.BF_sd('feat_key',simulation,feature_dict)
        feat_obj.initialize()
        layer_average = feat_obj.calculate_feature()[0]
        error = feat_obj.calculate_feature()[1]
        #score = feat_obj.calculate()
        #scores['BF8']['{}'.format(c)]=score

        av_burst_freq_8.append(layer_average) 
        error_8.append(error)
        sinu_freq_8.append(freq)

    negative_errors_6Hz =[x - y for x, y in zip(av_burst_freq_6, error_6)]
    positive_errors_6Hz = [x + y for x, y in zip(av_burst_freq_6, error_6)]
    positive_errors_8Hz = [x + y for x, y in zip(av_burst_freq_8, error_8)]
    negative_errors_8Hz = [x - y for x, y in zip(av_burst_freq_8, error_8)]

    if show:
        plt.figure()
        plt.suptitle(file_name)
        plt.plot(sinu_freq_6, av_burst_freq_6,label='6-pA burst freq.', color='teal')
        plt.plot(sinu_freq_6Hz, target_values_6Hz, label ='6-pA amplitude in-vitro recordings',color='teal', marker='o',linestyle='None')
        plt.fill_between(
            sinu_freq_6,
            negative_errors_6Hz,
            positive_errors_6Hz,
            color='teal',
            #label='firstcycles standard deviation 6 pA',
            linestyle='None',
            alpha=0.3)
        plt.fill_between(
            sinu_freq_8,
            negative_errors_8Hz,
            positive_errors_8Hz,
            color='orchid',
            #label='firstcycles standard deviation 8 pA',
            linestyle='None',
            alpha=0.3)
        plt.plot(sinu_freq_8, av_burst_freq_8,label='8-pA burst freq.',color='orchid')
        plt.plot(sinu_freq_8Hz, target_values_8Hz, label ='8-pA amplitude in-vitro recordings',color='orchid', marker='o',linestyle='None')

        plt.legend(fontsize='x-small')
        plt.ylabel('Av. burst frequency (Hz)')
        plt.xlabel('Sinusoidal stimulation frequency (Hz)')
        plt.ylim(0,100)
        #plt.xlim(0,25.5)
        #plt.xlim(0, max(sine_current_frequency)+2)
        if savefig:
            plt.savefig('{}_spiking_resonance.eps'.format(file_name))
            print('Figure of spiking resonance saved')
        plt.show()

     #### REPRESENTATION OF BURSTS  ####
    #------------------------------------------------------------
    # During sinusoidal current injections
    #------------------------------------------------------------
    #Definition of the sinusoidal current injection
    av_burst_freq_6 = []
    sinu_freq_6 = []
    error_6 = []
    sine_current_frequency=numpy.array([1.,6.,10.])
    init_time = 2. 
    end_time = 4.
    
    plt.figure()
    for c,freq in enumerate(sine_current_frequency): 
        simulation = new_initialisation(config_file,simulation_time,dict_name['param_configuration'])
        simulation.config_options['sin_current{}'.format(c)] = dict()
        simulation.config_options['sin_current{}'.format(c)]['amplitude'] = 6.0e-12 #A
        simulation.config_options['sin_current{}'.format(c)]['phase'] = 270 #degrees
        simulation.config_options['sin_current{}'.format(c)]['offset'] = 12.0e-12 # A
        simulation.config_options['sin_current{}'.format(c)]['frequency'] = freq
        simulation.initialize()
        #print nest.GetStatus(simulation.cerebellum.ac_generator[0].tolist())    
        simulation.run_simulation()
        if show:
            vm_time, vm_cell_id, vm = simulation.cerebellum.get_state_variable(state_variable='Vm', neuron_layer = 'mflayer',init_time=init_time,end_time=end_time)
            plt.subplot('31{}'.format(c+1))
            plt.plot(vm_time,vm*1e3, label = '{} Hz'.format(freq))
            plt.ylabel('Vm (mV)')
            plt.xlabel('Simulation time (s)')
            plt.legend()
            plt.ylim(-100,20)
    plt.tight_layout()
    if savefig:
        plt.savefig('{}_bursts_plot.eps'.format(file_name))
        print('Figure of bursts saved')
    plt.show()

    #### REPRESENTATION OF I-F AND FIRST-SPIKE LATENCY CURVES  ####
    #------------------------------------------------------------
    # During step-current injections
    #------------------------------------------------------------    
    step_current_amplitude = range(0,26,1)
    init_time= 0.
    end_time = 22.5
    simulation_time = 22.5
    pulse_amp = numpy.array([])
    neuron_freq = numpy.array([])
    neuron_lat = numpy.array([])
    #neuron_ai = numpy.array([])
    for c,amp in enumerate(step_current_amplitude):
        simulation = new_initialisation(config_file,simulation_time,dict_name['param_configuration'])
        simulation.config_options['pulse_current1'] = dict()
        simulation.config_options['pulse_current1']['init'] = init_time
        simulation.config_options['pulse_current1']['end'] = end_time
        simulation.config_options['pulse_current1']['amplitude'] = amp*1.e-12 # (A)
        simulation.initialize()
        simulation.run_simulation()

        #MEAN FREQUENCY FEATURE
        feature_dict = dict()
        feature_dict['layer'] = 'mflayer'
        feature_dict['stimulation'] = 'pulse_current1'
        feat_obj = MeanFrequency.MeanFrequency('feat_key',simulation,feature_dict)
        feat_obj.initialize()
        meanfreq = feat_obj.calculate_feature()

        #LATENCY FEATURE
        feature_dict = dict()
        feature_dict['layer'] = 'mflayer'
        feature_dict['stimulation'] = 'pulse_current1'
        feat_obj = Latency.Latency('feat_key',simulation,feature_dict)
        feat_obj.initialize()
        latency = feat_obj.calculate_feature()

        if meanfreq == 0.: 
            meanfreq = numpy.nan
        if latency == (end_time-init_time):
            latency = numpy.nan
        #if ai == end_time:
        #    ai = numpy.nan

        neuron_freq = numpy.append(neuron_freq,meanfreq)
        neuron_lat = numpy.append(neuron_lat,latency)
        #neuron_ai = numpy.append(neuron_ai,ai)
        pulse_amp = numpy.append(pulse_amp,amp)

    neuron_freq2 = neuron_freq[numpy.logical_not(numpy.isnan(neuron_freq))]
    pulse_amp2 = pulse_amp[numpy.logical_not(numpy.isnan(neuron_freq))]

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(pulse_amp2,neuron_freq2)
        if show:
            plt.figure(figsize=(10,5))
            #plt.subplot(131)
            plt.subplot(121)
            plt.plot(pulse_amp2,neuron_freq2, label='slope={}(Hz/pA), \n intercept={}pA, \n r_value={}, \n intercept2 = {}'.format(round(slope,2), round(intercept,2), r_value, pulse_amp2[0]))
            plt.plot(pulse_amp2, intercept + slope*pulse_amp2, 'r', label='fitted line')
            plt.plot((10,16,22),(30,45,60),marker='o',c='black',linestyle='None')
            plt.xlim(0,25)
            plt.ylim(0,100)
            plt.legend()
            plt.ylabel('Mean frequency (Hz)')
            plt.xlabel('Step current amplitude (pA)')
            #plt.title('{}-{}pA from {}-{}ms'.format(pulse_amp2[0],pulse_amp2[-1],init_time*1.e3, end_time*1.e3 ))

            #plt.subplot(132)
            plt.subplot(122)
            #plt.title('from {}-{}ms'.format(init_time*1.e3, end_time*1.e3 ))
            plt.plot(pulse_amp,numpy.array(neuron_lat)*1.e3, label='first-spike latency with {} pA = {}ms \n first-spike latency with {} pA = {}ms'.format(pulse_amp[~numpy.isnan(neuron_lat)][0], neuron_lat[~numpy.isnan(neuron_lat)][0]*1e3, pulse_amp[-1],neuron_lat[-1]*1e3))
            plt.plot((10,16,22),(31.9,19,14.65),marker='o',c='black',linestyle='None')
            plt.ylabel('First-spike latency (ms)')
            plt.xlabel('Step current amplitude (pA)')
            plt.legend()
            plt.xlim(0,25)
            plt.ylim(0,100)

            plt.tight_layout()
            if savefig:
                plt.savefig('{}_if_latency.eps'.format(file_name))
                print('Figure of I-F and first-spike latency curves saved')
            plt.show()

    except:
        slope=0
        print "There is no spike generated to calculate the I-F slope"
    
    

    
