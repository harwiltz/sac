import gym

from sac import SACAgent

agent = SACAgent(lambda : gym.make('CartPole-v0'),
                 hidden_size=128,
                 actor_lr=1e-4,
                 critic_lr=3e-4,
                 value_delay=1,
                 tau=5e-3,
                 max_replay_capacity=20000,
                 batch_size=64)

last_score = None

def visualize(artifacts):
    global last_score
    if artifacts['done']:
        last_score = artifacts['return']
#        print("[D] Episode {} score: {}".format(artifacts['episode'], artifacts['return']))
    step = artifacts['step']
    if step % 10000 == 0:
        critic_loss = 0.5 * (artifacts['loss']['critic1'] + artifacts['loss']['critic2'])
        actor_loss = artifacts['loss']['actor']
        value_loss = artifacts['loss']['value']
        fmt = "Step {:>5}| Actor Loss: {:>8.4f} | Critic Loss: {:>8.4f} | Value Loss: {:>8.4f} | Last Score: {}"
        print(fmt.format(step, actor_loss, critic_loss, value_loss, last_score))

def watch_demo(agent):
    env = gym.make('CartPole-v0')
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

agent.train(100000, visualizer=visualize)

while True:
    input("Press ENTER to view demo...")
    watch_demo(agent)
