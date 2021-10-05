#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A quick and terminal view over the results
from the cluster.
"""

import sys
import pickle
from logging_settings import logger

if __name__ == '__main__':
    file_dir = str(sys.argv[1])

with open(file_dir, 'rb') as fh:
    data_res = pickle.load(fh)

for index, p_res in enumerate(data_res['priors']):
    logger.info(f'SHD for round {index} is {p_res["SHD"]}')
