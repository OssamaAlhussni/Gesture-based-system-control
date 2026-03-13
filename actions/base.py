# actions/base.py
import time

class Action:
    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self._last_trigger = 0.0

    def can_trigger(self):
        return (time.time() - self._last_trigger) >= self.cooldown

    def trigger(self):
        if not self.can_trigger():
            return
        self._last_trigger = time.time()
        self.run()

    def run(self):
        raise NotImplementedError("Action must implement run()")
