import SpikingSimulation.Features.MeanFrequency
import SpikingSimulation.Features.Latency
import SpikingSimulation.Features.AdaptationIndex
import SpikingSimulation.Features.BurstFrequency
import SpikingSimulation.Features.ViabilityScanning
import SpikingSimulation.Features.SAPS
import SpikingSimulation.Features.BF_sd

def NC_Create_Feature_Translator_Dict():

	featureTranslatorDict = {
        	'mean_frequency':   SpikingSimulation.Features.MeanFrequency.MeanFrequency,
        	'latency':   SpikingSimulation.Features.Latency.Latency,
        	'adaptation_index':   SpikingSimulation.Features.AdaptationIndex.AdaptationIndex,
        	'burst_frequency':  SpikingSimulation.Features.BurstFrequency.BurstFrequency,
        	'viability_scanning': SpikingSimulation.Features.ViabilityScanning.ViabilityScanning,
        	'SAPS': SpikingSimulation.Features.SAPS.SAPS,
        	'BF_sd': SpikingSimulation.Features.BF_sd.BF_sd
    	}

	return featureTranslatorDict

def NC_Create_Parameter_Dic(config_options):
	parameter_keys = [key for key in config_options.keys() if key.startswith('parameter')]
        parameter_dic = []

        for key in parameter_keys:
            parameter_dic.append(config_options.pop(key))

	return parameter_dic

def NC_Create_StimulTranslatorList():
	stimulationTranslatorList = [
        	'pulse_current',
        	'sin_current'
    	]

	return stimulationTranslatorList
