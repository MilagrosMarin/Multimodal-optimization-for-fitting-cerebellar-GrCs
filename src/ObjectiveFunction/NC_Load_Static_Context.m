function [pyContext] = NC_Load_Static_Context(fileName)
    pyNestConfig = py.SpikingSimulation.Utils.Utils.ReadConfigFile(fileName);
    pyParameterDic = py.NC_Context_Builder.NC_Create_Parameter_Dic(pyNestConfig);
    pyFeatureTranslator = py.NC_Context_Builder.NC_Create_Feature_Translator_Dict();
    pyStimTransList = py.NC_Context_Builder.NC_Create_StimulTranslatorList();
    
    pyContext.pyNestConfig = pyNestConfig;
    pyContext.pyParameterDic = pyParameterDic;
    pyContext.pyFeatureTranslator = pyFeatureTranslator;
    pyContext.pyStimTransList = pyStimTransList;
end
