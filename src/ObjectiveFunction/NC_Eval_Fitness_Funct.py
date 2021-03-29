import SpikingSimulation.CurrentSimulation
import copy

def NC_Eval_Fitness_Funct(simulation_options, featureTranslatorDict, stimulationTranslatorList, parameter_dic, individual, seed):
        
        # Make a copy of the simulation config options
        local_config_options = copy.deepcopy(simulation_options)

        # Obtain a list with all the features in the configuration options
        feature_dict = {}
        stim_list = []
        for sec,val in local_config_options.items():
            for feat in featureTranslatorDict:
                if sec.startswith(feat):
                    feature_dict[sec] = val
                    local_config_options.pop(sec)
                    if val['stimulation'] not in stim_list:
                        stim_list.append(val['stimulation'])
                    break


        # Obtain the dict with all the stimulation in the configuration options
        stimulation_dict = {}
        for sec, val in local_config_options.items():
            for stim in stimulationTranslatorList:
                if sec.startswith(stim):
                    stimulation_dict[sec] = val
                    local_config_options.pop(sec)
                    break

        acumulated_score = 0.0

        # Simulate the network/neuron with each different stimulation in the list
        for stim in stim_list:
            stim_config_options = copy.deepcopy(local_config_options)
            stim_config_options[stim] = stimulation_dict[stim]

            # Obtain the list of features related to this stimulation
            feat_to_calculate = {}
            for name,val in feature_dict.items():
                if val['stimulation']==stim:
                    feat_to_calculate[name] = val

            unnorm_values = individual #self._get_unnormalized_values(individual)
        
            for unnorm, param_dic in zip(unnorm_values, parameter_dic):
                stim_config_options[param_dic['section']][param_dic['parameter']] = unnorm

            stim_config_options['simulation']['seed'] = seed

            feature_values = helper_simulation(stim_config_options, featureTranslatorDict, feat_to_calculate) #INICIAL: helper_simulation(stim_config_options, feat_to_calculate)

            for val in feature_values:
                acumulated_score += val

        return acumulated_score

######### Funcion de evaluacion auxiliar inyectada:

def helper_simulation(local_config_options, featureTranslatorDict, feat_to_calculate):
    try:
        simulation = SpikingSimulation.CurrentSimulation.CurrentSimulation(config_options = local_config_options)

        simulation.initialize()

        simulation.run_simulation()

        score_list = []
        for feat_key,feat_dict in feat_to_calculate.items():
            for feat_name,feat_class in featureTranslatorDict.items():
                if feat_key.startswith(feat_name):
                    feat_obj = feat_class(feat_key,simulation,feat_dict)
                    feat_obj.initialize()
                    score = feat_obj.calculate()
                    score_list.append(score)

    except KeyboardInterrupt:
        print('Received SIGNINT signal. Ending simulation')
        import sys
        sys.exit(0)

    except Exception as err:
        score_list = []
        for feat_key, feat_dict in feat_to_calculate.items():
            for feat_name, feat_class in featureTranslatorDict.items():
                if feat_key.startswith(feat_name):
                    feat_obj = feat_class(feat_key, simulation, feat_dict)
                    feat_obj.initialize()
                    score = feat_obj.max_value()
                    score_list.append(score)
         
    return score_list
