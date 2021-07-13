import pickle
# Random policy
class rand_policy():
    def __init__(self, env):
        self.action_space = env.action_space # Storing env makes in unpickable
    def get_action(self, obs):
        return list([self.action_space.sample(), 0.0])

def get_policy(env, policy):
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = rand_policy(env)
    return pi
    
