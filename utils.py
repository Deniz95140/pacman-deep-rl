import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.monitor import Monitor

def make_env(render_mode=None, seed: int = 0):
    """Construit MsPacman avec pré-traitements Atari.
    - Grayscale 84x84
    - Frame-skip=4 (géré par le wrapper; ALE est à frameskip=1)
    - Terminal sur perte de vie
    - Empilement de 4 frames (C, H, W)
    """
    # Important: ale_py doit être importé/registré avant gym.make dans l'appelant si besoin
    env = gym.make("ALE/MsPacman-v5", render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        terminal_on_life_loss=True,
    )
    env = FrameStackObservation(env, stack_size=4, padding_type="reset")
    env = Monitor(env)
    return env
