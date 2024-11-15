# Audio Imports
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class Audio:
    def __init__(self):
        # Initialize Pycaw for volume control.
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = interface.QueryInterface(IAudioEndpointVolume)

    def set_volume(self, value: float):
        self.volume.SetMasterVolumeLevelScalar(value, None)

    def get_volume(self) -> int:
        return round(self.volume.GetMasterVolumeLevelScalar() * 100)
