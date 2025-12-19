import numpy as np

def policy_buy_hold(obs):
    # always 100%
    return 3

def policy_flat(obs):
    return 0

def policy_avoid_earnings(obs):
    # obs[-2] = earnings flag per our obs vector
    eflag = obs[-2]
    return 0 if eflag >= 0.5 else 3
