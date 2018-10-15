import configparser
import os

from jinja2 import Template


CONFIG = configparser.ConfigParser()

if os.path.exists(os.path.expanduser('~/.metric_learning.conf')):
    CONFIG.read(os.path.expanduser('~/.metric_learning.conf'))
else:
    conf_path = '{}/../default.conf'.format(os.path.dirname(__file__))
    CONFIG.read(conf_path)


def render_jinja_config(config_name, **kwargs):
    file_path = '{}/../metric_learning/configurations/{}.jinja2'.format(
        os.path.dirname(__file__), config_name)
    with open(file_path) as f:
        template = Template(f.read())
    return eval(template.render(**kwargs))


def generate_config(parameters):
    return {
        x: render_jinja_config(x, **y) for x, y in parameters
    }