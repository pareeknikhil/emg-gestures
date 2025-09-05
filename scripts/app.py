
from src.utils.data_processing import load_env_variables

load_env_variables()

from src.utils.log_utils import Logger
from src.visualizer.window import EMGSignalAnalyzer


class App(EMGSignalAnalyzer):
    pass

if __name__=="__main__":
    logger = Logger.get_instance()
    App.run(logger)
