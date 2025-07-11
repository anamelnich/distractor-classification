# logger.py
import Pd_config as config
_current_logger = None

def set_current_logger(logger):
    """Call this once in main.py to register your TrialLogger."""
    global _current_logger
    _current_logger = logger

def get_current_logger():
    """Returns the active TrialLogger, or None if none is set."""
    return _current_logger

class TrialLogger:
    def __init__(self, basename: str | None = None):
        if not basename:
            basename = datetime.now().strftime("test_%Y%m%d%H%M%S")
        self.basename      = basename
        self._lines_analyze = []
        self._triggers      = []

    def log_trial(self, trial_idx, task, feedback, tpos, dpos, dot_correct, BCI_output=None):
        if config.MODE != "decode" or BCI_output is None:
            BCI_output = 99
        self._lines_analyze.append(
            f"{trial_idx+1} {task} {feedback} {tpos} {dpos} {dot_correct} {BCI_output}"
        )

    def log_trigger(self, code, timestamp, trial_idx):
        self._triggers.append((code, timestamp, trial_idx))

    def dump(self):
        with open(f"{self.basename}.analysis.txt", "w") as f:
            f.write("\n".join(self._lines_analyze) + "\n")
        with open(f"{self.basename}.triggers.txt", "w") as f:
            for code, ts, ti in self._triggers:
                f.write(f"{ti+1} {code} {ts}\n")



