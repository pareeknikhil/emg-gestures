import sys
import time

from src.utils.data_processing import load_env_variables

load_env_variables()

from src.pipeline.combineTFR import (combine_labels,
                                     print_combine_sample_collected)
from src.pipeline.deleteTFR import delete_old_files
from src.pipeline.emg2vec import run_clr
from src.pipeline.gesture_embedding import run_model
from src.pipeline.labelTFR import create_window, print_sample_collected
from src.pipeline.rae_embedding import run_embedding
from src.pipeline.scale import print_scaled_sample_collected, scale
from src.utils import \
    gpu_config  # # [TECH DEBT: add a method instead of script-run]

selected_type = sys.argv[1]

if selected_type == "model":
    run_model()
elif selected_type == "embedding":
    run_embedding()
elif selected_type == 'clr':
    run_clr()
else:
    delete_old_files(selected_type=[selected_type]) ## [TECH DEBT: add different folder for combined EMG (removes circular dependency)]

    time.sleep(2)

    create_window(selected_type=selected_type)

    print_sample_collected(selected_type=selected_type)

    time.sleep(2)

    combine_labels(selected_type=selected_type)

    print_combine_sample_collected(selected_type=selected_type)

    scale(selected_type=selected_type)

    print_scaled_sample_collected(selected_type=selected_type)
