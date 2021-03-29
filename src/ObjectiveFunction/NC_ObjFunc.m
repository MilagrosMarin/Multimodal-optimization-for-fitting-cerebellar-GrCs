function fitness = NC_ObjFunc(candidate_solution, context)
    fitness = py.NC_Eval_Fitness_Funct.NC_Eval_Fitness_Funct(context.pyNestConfig, ...
        context.pyFeatureTranslator, context.pyStimTransList, context.pyParameterDic, ...
        py.list(candidate_solution), context.seed);
end