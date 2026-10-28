"""Multi-robot indoor target search & tracking simulation.

Infrastructure modules (complete): environment, robot, target, planning,
faults, dispatch (pipeline), harness, visualize.

Research-team modules (STUBS ONLY, see README): strategies, comms, and the
``select_robot_for_dispatch`` function in dispatch.
"""
from .environment import Environment
from .robot import Robot
from .target import Target
from .planning import plan_path
from .harness import SimConfig, Simulation, run_trials

__all__ = ["Environment", "Robot", "Target", "plan_path", "SimConfig", "Simulation", "run_trials"]
