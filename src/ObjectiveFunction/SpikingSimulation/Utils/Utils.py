import ConfigParser
import argparse


def ReadConfigFile(config_file):
    '''
    Read all the sections in the configuration file.
    @param config_file Configuration file to be readed
    '''
    config_parser = ConfigParser.ConfigParser()
    config_parser.read(config_file)
    
    # Read every section in the file
    sections = config_parser.sections()
    config_options = dict()
    for sect in sections:
        config_options[sect] = ConfigSectionMap(config_parser = config_parser, section = sect)
    
    return config_options
    
def ReadConfigParameters(argv):
    
    config_options = dict()
    
    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    if args.conf_file:
        config_options['config_file'] = args.conf_file
    
    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
        )
    
    parser.add_argument('options', action='append', metavar='option', type=name_value_pair, nargs='*',
                   help='option=value')
    args = parser.parse_args(remaining_argv)
    
    if args.options:
        for option in args.options[0]:
            if option['section'] not in config_options:
                config_options[option['section']] = dict()
            config_options[option['section']][option['parameter']] = option['value']
    
    return config_options

def ExpandConfigParameters(**kwargs):
    '''
    It expands the configuration options from dict[section.key]=value to dict[section][key]=value 
    '''
    
    config_options = dict()
    
    for id_param in kwargs:
        param = id_param.split('.')
        if len(param)!=2:
            msg = "%r invalid section.param pattern" % id_param
            raise argparse.ArgumentTypeError(msg)
        
        if param[0] not in config_options:
            config_options[param[0]] = dict()
        config_options[param[0]][param[1]] = kwargs[id_param]
        
    return config_options

def name_value_pair(string):
    value = string.split('=')
    if len(value)!=2:
        msg = "%r invalid section.param=value pattern" % string
        raise argparse.ArgumentTypeError(msg)
    
    param = value[0].split('.')
    if len(param)!=2:
        msg = "%r invalid section.param=value pattern" % string
        raise argparse.ArgumentTypeError(msg)
    
    retvalue = {'section': param[0],
                'parameter': param[1],
                'value': parseOption(value[1])
                }
    
    return retvalue
    
    

def WriteConfigFile(config_dict, file_name):
    '''
    Write the simulation configuration into a file
    '''
    parser = ConfigParser.ConfigParser()
    
    # Add every section to the file
    for sec in config_dict.keys():
        parser.add_section(sec)
        
        for key in config_dict[sec].keys():
            parser.set(sec, key, config_dict[sec][key])

    with open(file_name, 'w') as f:
        parser.write(f)
        
def ConfigSectionMap(config_parser, section):
    '''
    This function extracts all the properties in a specific section of the file.
    '''
    dict1 = {"name": section}
    options = config_parser.options(section)
    for option in options:
        try:
            dict1[option] = parseOption(config_parser.get(section, option))
        except:
            print 'exception on ',option
            dict1[option] = None
    return dict1
            
def boolify(s):
    if s == 'True' or s == 'true':
            return True
    if s == 'False' or s == 'false':
            return False
    raise ValueError('Not Boolean Value!')

def parseOption(value):
    var = value.split(',')
    new_var = [estimateType(val) for val in var]
        
    if len(new_var)==0:
        return None
    elif len(new_var)==1:
        return new_var[0]
    else:
        return new_var

def estimateType(var):
    '''guesses the str representation of the variables type'''
    var = str(var) #important if the parameters aren't strings...
    for caster in (boolify, int, float):
        try:
            return caster(var)
        except ValueError:
            pass
    
    return var