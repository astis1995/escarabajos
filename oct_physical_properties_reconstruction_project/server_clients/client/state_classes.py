from enum import Enum, auto

class AvantesConnectionState(Enum):
    CONNECTED = auto()
    DISCONNECTED = auto()



class LabscopeConnectionState(Enum):
    CONNECTED = auto()
    DISCONNECTED = auto()



class CurrentScopeState(Enum):
    OCT = auto()
    SPECTROMETER = auto()    