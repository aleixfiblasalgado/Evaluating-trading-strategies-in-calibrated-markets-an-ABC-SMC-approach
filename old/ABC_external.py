#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## Approximate Bayesian Computation
@author: aleix
"""

import pyabc
import pyabc.external


log_dir = "rmsc04"

market_simulation = pyabc.external.ExternalHandler(
    executable="bash", file="scripts/rmsc04.sh"
)

market_simulation.create_executable(loc=log_dir)

market_simulation.run()

