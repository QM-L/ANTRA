import matplotlib
matplotlib.use("QtAgg")

from antra.interface.interface import run_application
from antra.general.config import load_configs
from totalsegmentator.map_to_binary import class_map

if __name__ == "__main__":
    run_application()
