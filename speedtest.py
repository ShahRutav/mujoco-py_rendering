import gym
import click
import mjrl
import time
from termcolor import colored


# Random policy
class rand_policy():
    def __init__(self, env):
        self.env = env
    def get_action(self, obs):
        return [self.env.action_space.sample()]

def preliminary_test():
    print(colored("\n++++++++++++++++++ Running preliminary test using mujoco_py.cymj", "green"))
    import mujoco_py
    cymj = str(mujoco_py.cymj)
    if "gpu" in cymj : 
        print(colored("Found GPU extension.", "red"))
    elif "cpu" in cymj :
        print(colored("Found CPU extension.", "red"))
    else : 
        raise Exception

DESC = ""
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', default="mjrl_swimmer-v0")
@click.option('-e','--episodes', type=int, help='Number of episodes', default=10)
@click.option('-c','--camera', type=str, help='Camera name', default='cam0')
@click.option('-p','--policy', type=str, help='Path to policy', default=None)
def main(env_name, episodes, camera, policy):
    env = gym.make(env_name)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
        mode = 'exploration'
    else:
        pi = rand_policy(env)
        mode = 'exploration'
    width = 224
    height = 224
    total_rendering_time = 0.0
    total_steps = 0
    obs = env.reset() 
    for ep in range(episodes):
        
        o = env.reset()
        d = False
        t = 0
        score = 0.0
        while d is False:
            a = pi.get_action(o)[0] if mode == 'exploration' else pi.get_action(o)[1]['evaluation']
            o, r, d, _ = env.step(a)
            
            start_time = time.time()
            img = env.sim.render(width=width, height=height, mode='offscreen', camera_name=camera, device_id=0)
            total_rendering_time += time.time() - start_time
            img = img[::-1,:,:] # Image is flipped.

            t = t+1
            total_steps += 1
            score = score + r
    
    avg_rendering_time = total_rendering_time / total_steps
    print(colored("Average time to render one image : {}".format(avg_rendering_time), "red"))
    print(colored("Frequency : {} images/second".format(1/avg_rendering_time), "red"))


if __name__ == '__main__':
    preliminary_test()

    print(colored("\n++++++++++++++++++ Expected results on mjrl_swimmer-v0 environment (Averaged over 100 episodes). Tested on CPU : Intel Xeon Gold 5118 @ 2.30GHz; GPU : NVIDIA Quadro P4000; OS : CentOS 7", "green"))
    print(colored("\n##### For CPU based rendering ", "blue"))
    print(colored("Average time to render one image (224x224) ~  0.035 seconds, Frequency ~ 30 images/second", "blue"))
    print(colored("\n##### For GPU based rendering ", "blue"))
    print(colored("Average time to render one image (224x224) ~  0.0006 seconds, Frequency ~ 1500 images/second", "blue"))
    
    print(colored("\n++++++++++++++++++ Running speed test on your machine", "green"))

    main() 
