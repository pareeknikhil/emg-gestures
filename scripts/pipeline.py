import sys
import time

from src.utils.data_processing import load_env_variables

load_env_variables()

from src.pipeline.combineTFR import (combine_labels,
                                     print_combine_sample_collected)
from src.pipeline.deleteTFR import delete_old_files
from src.pipeline.inference import run_test
from src.pipeline.labelTFR import create_window, print_sample_collected
from src.pipeline.model import run_model
from src.utils.gpu_config import limit_gpu_memory

limit_gpu_memory()

selected_type = sys.argv[1]


if selected_type == "inference":
    run_test()
elif selected_type == "model":
    run_model()
else:
    delete_old_files(selected_type=[selected_type]) ## [TECH DEBT: add different folder fpr combined EMG (removes circular dependency)]

    time.sleep(2)

    create_window(selected_type=selected_type)

    print_sample_collected(selected_type=selected_type)

    time.sleep(2)

    combine_labels(selected_type=selected_type)

    print_combine_sample_collected(selected_type=selected_type)
