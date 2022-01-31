# agents
from .A2C import A2C_linear, A2C
from .CRPLSTM import CRPLSTM
from .GRU import GRU
from ._rl_helpers import compute_returns, compute_a2c_loss, pick_action
