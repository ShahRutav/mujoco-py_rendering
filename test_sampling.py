import gym
import mjrl
from termcolor import colored
import time
from multicam import PixelWrapper
from utils import get_policy
import click
import multiprocessing
from core import sample_paths 
from PIL import Image

DESC = ""
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', default="mjrl_swimmer-v0")
@click.option('-e','--episodes', type=int, help='Number of episodes', default=10)
@click.option('--num_proc', type=int, help='Number of processes to launch', default=2)
@click.option('-c','--cameras', type=str, help='Camera name', default=['cam0'], multiple=True)
@click.option('-p','--policy', type=str, help='Path to policy', default=None)
def main(env_name, episodes, cameras, policy, num_proc):
    width = 224
    height = 224
    env = gym.make(env_name)
    env = PixelWrapper(env, cameras)
    env_kwargs = {'env_name': env_name, 'PixelWrapper' : env.env_kwargs}
    pi = get_policy(env, policy)
    mode = 'exploration'

    paths = sample_paths(num_traj=episodes, env=env_name, policy=pi, env_kwargs=env_kwargs, num_cpu=num_proc, base_seed=123)
    #for path in paths:
    #    obs = path['observations']
    #    for step in range(obs.shape[0]) :
    #        img1 = obs[step, 0, :, :, :]
    #        img1 = Image.fromarray(img1)
    #        img1.save(f"img{step:02d}.png")
    #    break

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn') # Important to set for multiprocessing with GPU
	main()
