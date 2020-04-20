import gym
import hydra
import numpy as np

from sac import SACAgent

@hydra.main(config_path='config.yaml')
def main(cfg):
    agent = hydra.utils.instantiate(cfg.agent)
    agent.train(cfg.training.episodes)

    if cfg.training.preview:
        while True:
            input("Press ENTER to view demo...")
            rewards = agent.rollout(render=True)
            print("Episode score: {}".format(rewards))

if __name__ == "__main__":
    main()
