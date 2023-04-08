from utils.logger import make_logger
from utils.filehandling import ConfLoader
from models.modelhandling import load_model

logger = make_logger(__name__) 
conf = ConfLoader("config.yml")

class Detector:
    def __init__(self, modelpath, datapath):

        self.model = load_model(modelpath)
        self.spectrogram = np.load(datapath + "spectrogram.npy", allow_pickle=True)
        self.frequencies np.load(datapath + "frequencies.npy", allow_pickle=True)
        self.times = np.load(datapath + "times.npy", allow_pickle=True)
        self.traces = np.load(datapath + "traces.npy", allow_pickle=True)

        self.freq_pad = conf.freq_pad
        self.time_pad = conf.time_pad
        

    def detect(self, image):
        logger.info("Detecting...")
