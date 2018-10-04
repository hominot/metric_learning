import configparser
import os

config = configparser.ConfigParser()

if os.path.exists(os.path.expanduser('~/.metric_learning.conf')):
    config.read(os.path.expanduser('~/.metric_learning.conf'))
else:
    conf_path = '{}/../default.conf'.format(os.path.dirname(__file__))
    config.read(conf_path)
