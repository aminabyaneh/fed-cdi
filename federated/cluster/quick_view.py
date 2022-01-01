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
    
    n_clients = len(data_res.keys()) - 2
    
    for index, p_res in enumerate(data_res['priors']):
        avg_SHD = 0
        for c in range(n_clients):
            avg_SHD += data_res[c][index]["SHD"] / n_clients
        
        logger.info(f'SHD for round {index} is {p_res["SHD"]}, {avg_SHD}')

