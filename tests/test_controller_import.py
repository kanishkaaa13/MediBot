"""Exercise controller import, initialization, and 200 loop iterations without Webots."""
import importlib
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_DIR = os.path.join(ROOT, "controllers", "medibot_controller")
sys.path.insert(0, CONTROLLER_DIR)


class Device:
    def __init__(self, name=""):
        self.name = name
        self.value = 4.0 if name.startswith("us_") else 0.0

    def enable(self, *_args): pass
    def setPosition(self, *_args): pass
    def setVelocity(self, *_args): pass
    def set(self, *_args): pass
    def getValue(self): return self.value
    def getValues(self): return [0.0, 0.0, 0.0]
    def getImage(self): return None
    def getKey(self): return -1


class Field:
    def getCount(self): return 0
    def getMFNode(self, _index): return Node()
    def importMFNodeFromString(self, *_args): pass
    def setSFVec3f(self, *_args): pass
    def setSFRotation(self, *_args): pass


class Node:
    def __init__(self): self.fields = {}
    def getField(self, name): return self.fields.setdefault(name, Field())
    def getPosition(self): return [0.0, 0.0, 0.0]
    def getOrientation(self): return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    def resetPhysics(self): pass


class FakeSupervisor:
    def __init__(self):
        self.devices = {}
        self.node = Node()
        self.time = 0.0
        self.steps = 0

    def getDevice(self, name): return self.devices.setdefault(name, Device(name))
    def getKeyboard(self): return self.getDevice("keyboard")
    def getSelf(self): return self.node
    def getFromDef(self, _name): return Node()
    def getRoot(self): return self.node
    def getTime(self): return self.time
    def setLabel(self, *_args): pass
    def step(self, timestep):
        self.steps += 1
        self.time += timestep / 1000.0
        return 0 if self.steps <= 200 else -1
    def simulationQuit(self): pass


fake_controller = types.ModuleType("controller")
fake_controller.Supervisor = FakeSupervisor
sys.modules["controller"] = fake_controller
sys.modules.pop("medibot_controller", None)

controller_module = importlib.import_module("medibot_controller")
bot = controller_module.MediBot()
bot.route = ["HOME", "PHARMACY", "JUNCTION", "WARD-2A", "WARD-3B"]
bot.idx = 3
bot.ekf.x = [4.0, 0.0, 1.57]
placement = bot._point_ahead_on_route(1.0)
assert placement is not None and abs(placement[0] - 4.0) < 1e-9 and abs(placement[1] - 1.0) < 1e-9
assert placement[3] == 2
bot.run()
assert bot.robot.steps == 201, f"expected 201 total steps, got {bot.robot.steps}"
print("PASS  controller import, initialization, route-ahead trolley placement, and 200 simulation steps")
