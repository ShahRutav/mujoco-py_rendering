import gym
import mjrl
from termcolor import colored
import time
from multicam import PixelWrapper
from utils import get_policy
import click

DESC = ""
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', default="mjrl_swimmer-v0")
@click.option('-e','--episodes', type=int, help='Number of episodes', default=10)
@click.option('-c','--cameras', type=str, help='Camera name', default=['cam0'], multiple=True)
@click.option('-p','--policy', type=str, help='Path to policy', default=None)
def main(env_name, episodes, cameras, policy):
    width = 224
    height = 224
    env = gym.make(env_name)
    env = PixelWrapper(env, cameras)
    pi = get_policy(env, policy)
    mode = 'exploration'

    total_rendering_time = 0.0
    total_steps = 0
    obs = env.reset() 
    for ep in range(episodes):
        o = env.reset()
        total_steps += 1
        d = False
        score = 0.0
        while d is False:
            a = pi.get_action(o)[0] if mode == 'exploration' else pi.get_action(o)[1]['evaluation']
            start_time = time.time()
            o, r, d, _ = env.step(a)
            total_rendering_time += time.time() - start_time

            total_steps += 1
            score = score + r
    
    avg_rendering_time = total_rendering_time / (total_steps*len(cameras))
    print(colored("Note :", "red"), colored("Time decreases compared to speedtest.py since we are measuring both simulation + image rendering time.", "green"))
    print(colored("Average time to render one image : {}".format(avg_rendering_time), "red"))
    print(colored("Frequency : {} images/second".format(1/avg_rendering_time), "red"))

if __name__ == '__main__':
    main()
