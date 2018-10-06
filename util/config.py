import configparser
import os

CONFIG = configparser.ConfigParser()

if os.path.exists(os.path.expanduser('~/.metric_learning.conf')):
    CONFIG.read(os.path.expanduser('~/.metric_learning.conf'))
else:
    conf_path = '{}/../default.conf'.format(os.path.dirname(__file__))
    CONFIG.read(conf_path)
