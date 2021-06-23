import importlib_resources as _resources
import configparser


__VERSION_INFO = '1.1.0'

_config = configparser.ConfigParser()
with _resources.path("curvefitgui", "config.txt") as _path:
    _config.read(str(_path))

settings = {}

# general
settings['MODEL_NUMPOINTS'] = int(_config['general']['numpoints'])
settings['SIGNIFICANT_DIGITS'] = int(_config['general']['significant_digits'])
settings['XERRORWARNING'] = _config.getboolean('general','show_x_error_warning')
settings['SORT_RESIDUALS'] = _config.getboolean('general','sort_residuals')
    
# fitparameters
settings['CM_SIG_DIGITS'] = int(_config['fitparameter']['significant_digits'])
settings['CM_SIG_DIGITS_NO_ERROR'] = int(_config['fitparameter']['significant_digits_fixed'])

# reportview
settings['REPORT_FONT'] = _config['reportview']['font']
settings['REPORT_SIZE'] = int(_config['reportview']['size'])

# ticklabels
settings['TICK_COLOR'] = _config['ticklabels']['color']
settings['TICK_FONT'] = _config['ticklabels']['font']
settings['TICK_SIZE'] = int(_config['ticklabels']['size'])

# text
settings['TEXT_FONT'] = _config['text']['font']
settings['TEXT_SIZE'] = int(_config['text']['size'])

# errorbars
settings['BAR_Y_COLOR'] = _config['errorbars']['y_bar_color']
settings['BAR_X_COLOR'] = _config['errorbars']['x_bar_color']
settings['BAR_Y_THICKNESS'] = int(_config['errorbars']['y_bar_thickness'])
settings['BAR_X_THICKNESS'] = int(_config['errorbars']['x_bar_thickness'])

