#!/bin/bash

python get_config_names.py | while read config; do python run.py --config=$config; done
