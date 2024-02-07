import warnings
import dreamerv3
from dreamerv3 import embodied
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import crafter
from embodied.envs import from_gym
import gym#nasium as gym
from dreamerv3.embodied.core.wrappers import CustomResizeImage
from gym_chess import ChessEnvV1, ChessEnvV2  # necessary to register the chess envs

# MOD: Adjust flags manually based on preference.
game = 'chess_gym'  # 'crafter', 'chess_gym', 'lunar_lander', or any other supported game.
run_count = '1'  # Used to train different algos separately.
resize_obs_space = (64,) # 'None', int, or 1D/2D/3D tuple. Use multiples of 2, see here: https://github.com/danijar/dreamerv3/issues/12
do_eval = True  # 'False' or 'True' (train vs. evaluate)
do_export_actions = True  # 'False' or 'True', export actions to external file (e.g. to play a game in real time)

def main():
  logdir = f'~/logdir/{game}-{run_count}'
  checkpoint = logdir + '/checkpoint.ckpt'
  checkpoint = checkpoint if embodied.Path(checkpoint).exists() else None

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs[game])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': logdir,
      # 'run.train_ratio': 64,
      # 'run.log_every': 30,  # Seconds
      # 'batch_size': 16,
      # 'jax.prealloc': False,
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  if game == 'crafter':
    env = crafter.Env()  # Replace this with your Gym env.
  elif game == 'lunar_lander':
    env = gym.make('LunarLander-v2', render_mode='human')
  elif game == 'chess_gym':
    # env = gym.make('ALE/VideoChess-v5', obs_type='rgb', render_mode='human' if do_eval else None)
    env = gym.make('ChessVsSelf-v1')
    if resize_obs_space: env = CustomResizeImage(env, resize_obs_space)
  
  env = from_gym.FromGym(env, obs_key='vector')  # 'image' (Crafter, Chess) or 'vector' (LunarLander)
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config({
    **config.run,
    'logdir': config.logdir,
    'batch_steps': config.batch_size * config.batch_length,
    # 'plt_render': do_eval and game == 'crafter',  # 'render_mode' not working in Crafter, we work around w/ this bool
    'from_checkpoint': checkpoint,
    'export_actions': do_export_actions,
    # 'truncate': 30000,
  })
  
  if not do_eval:
    embodied.run.train(agent, env, replay, logger, args)
  else:
    embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
