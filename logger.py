import logging as lg
import os

class Log:
    def __init__(self):
        try:
            self.logFile="ai4i2020.log"
            
            # removing the log file if already exists so as not to congest it.
            if os.path.exists(self.logFile):
                os.remove(self.logFile)
            lg.basicConfig(filename=self.logFile, level=lg.INFO, format="%(asctime)s %(levelname)s %(message)s")
            
            # Adding the StreamHandler to record logs in the console.
            self.console_log = lg.StreamHandler()
            
            # setting level to the console log.
            self.console_log.setLevel(lg.INFO) 
            
            # defining format for the console log.
            self.format = lg.Formatter("%(levelname)s %(asctime)s %(message)s")
            self.console_log.setFormatter(self.format) 
            
            # adding handler to the console log.
            lg.getLogger('').addHandler(self.console_log) 
        
        except Exception as e:
            lg.info(e)
            
        else:
            lg.info("Log Class successfully executed!")