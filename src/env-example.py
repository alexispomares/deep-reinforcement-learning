import gym
import random
from gym_chess import ChessEnvV1
import numpy as np

def main():

  env = ChessEnvV1()
  # env = gym.make("ChessVsSelf-v1") # equivalent method

  env.reset()

  for _ in range(1000):
    # # select a move and convert it into an action
    # moves = env.possible_moves
    # move = random.choice(moves)
    # action = env.move_to_action(move)

    # or select an action directly
    actions = env.possible_actions
    action = random.choice(actions)

    observation, reward, done, info = env.step(action)
    # print('Obs. Info:', observation, reward, done, info, '\n')

    if done:
      state = env.reset()
      print('Done!', state)
      break

  env.close()



if __name__ == '__main__':
  main()