"""Run an experiment sweep. See src/qkdsim/experiment.py for the config format.

    python experiments/run_experiment.py experiments/configs/e2_mesh_load.yaml [--quick] [--dry-run] [--workers N]
"""

import sys

from qkdsim.experiment import main

if __name__ == "__main__":  # required on Windows: worker processes re-import this file
    sys.exit(main())
