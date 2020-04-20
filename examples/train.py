import gym
import numpy as np

from sac import SACAgent

agent = SACAgent(lambda: gym.make('LunarLanderContinuous-v2'), value_delay=2)

def visualize(artifacts):
    if artifacts['done']:
        if artifacts['episode'] % 10 == 0:
            print("=" * 65)
            results = agent.rollout(10)
            mean = np.mean(results)
            _min = np.min(results)
            _max = np.max(results)
            fmt = "Ep {:>4} | Mean: {:>8.2f} | Min: {:>8.2f} | Max: {:>8.2f}"
            print(fmt.format(artifacts['episode'], mean, _min, _max))
            print("=" * 65)
    step = artifacts['step']
    if step % 1000 == 0:
        critic_loss = 0.5 * (artifacts['loss']['critic1'] + artifacts['loss']['critic2'])
        actor_loss = artifacts['loss']['actor']
        value_loss = artifacts['loss']['value']
        fmt = "Step {:>5}| Actor Loss: {:>8.4f} | Critic Loss: {:>8.4f} | Value Loss: {:>8.4f}"
        print(fmt.format(step, actor_loss, critic_loss, value_loss))

def watch_demo(agent):
    env = gym.make('LunarLanderContinuous-v2')
    done = False
    score = 0
    s = env.reset()
    while not done:
        a = agent.action(s)
        s, r, done, _ = env.step(a)
        score += r
        env.render()
    env.close()
    print("Score: {}".format(score))

agent.train(80000, visualizer=visualize)

while True:
    input("Press ENTER to view demo...")
    watch_demo(agent)
