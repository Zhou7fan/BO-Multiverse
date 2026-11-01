import os

PROGRESS_BAR_LEN = 30


def experiment_directory():
    # Patched from original COMBO/config.py (which raised on unknown hostnames).
    # All COMBO artifacts land under engine/bo/logs/combo_exp/.
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'combo_exp')
    os.makedirs(d, exist_ok=True)
    return d
