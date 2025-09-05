import logging


class Logger:
    __instance = None

    @classmethod 
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    @staticmethod
    def get_handler():        
        handler = logging.StreamHandler()
        _format = "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
        _formatter = logging.Formatter(_format)
        handler.setFormatter(_formatter)
        return handler

    def __init__(self) -> None:
        self.logger = logging.getLogger("EMgGUI")
        self.logger.setLevel(logging.NOTSET)
        self.logger.addHandler(Logger.get_handler())
    
    def info(self, message):
        self.logger.info(msg=message)
    
    def error(self, message):
        self.logger.error(msg=message)
    
    def release(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()
        self.logger = None
        Logger.__instance = None
        print("Logger resources released successfully.")