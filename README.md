# Exploring the SOTA in Deep Reinforcement Learning

## Motivation

This is a work in progress (WIP) to eventually build a general-purpose Deep Reinforcement Learning (RL) pipeline for algorithms that can learn to play any real-time game from scratch, and ideally **reach top-human or superhuman level using only the raw screen pixels as input**. This is a somewhat ambitious goal that will likely not be fulfilled for a few years still, but it's also an important one because it would mean that we have finally have at our disposal a general-purpose AI that can learn tasks with:

- The same exact input than a human, and...
- ...a similar amount of training data (rather than many orders of magnitude more, like in traditional RL).

Our main steps along the way will be:

1. Firstly, have a model that can learn to play chess (a relatively complex but perfect-information environment) at competitive level.
2. Once we will have mastered chess, apply a similar approach to imperfect-information games. In our case it will be Clash Royale.
3. The final step will be having a **highly generalizable RL architecture** (based on DreamerV3, which requires close-to-no fine-tuning for high performance) that learns new tasks within a **reasonable timeframe and compute resources** (e.g. the DreamerV3 authors trained the algo to perform a very complex task in Minecraft using **a single GPU** to simulate only **17 days of playtime**).

In my opinion, based on the evidence from large players in the industry (OpenAI, DeepMind, and many others) this is a directionally correct approach towards ASI (Artificial Super-Intelligence) that can generate truly novel insights and applications; in contrast, we may already achieve AGI (Artificial General Intelligence) with a sufficiently smart LLM-powered OS that can generalize to a multitude of different real-world tasks where we have lots of data at our disposal. The key magic ingredient with Reinforcement Learning is that **it allows to transform compute power into data** (by accumulating action-observation loops), while LLMs require colossal amounts of supervised training data to perform near human level — sometimes even forcing us to create synthetic training data — and they still have significantly limiting drawbacks such as inherent hallucinations.

TL;DR — AGI may be achieved by scaling up the compute thrown at LLMs, but ASI will likely only be achieved with some sort of (highly data-efficient) hybrid variation that includes Deep Reinforcement Learning — perhaps similar in concept to the [model-based Dreamer architecture](https://arxiv.org/abs/2301.04104) which navigates the world by first learning a simplified model of it, much like humans do!

![DreamerV3 Tasks](/img/dreamerv3/intro.gif)

## Tech Stack

The general architecture includes:

- TODO: elaborate
<!--- - [Dreamer V3](https://danijar.com/project/dreamerv3) for the Deep RL algo, regarded as the [current state of the art](https://reddit.com/r/MachineLearning/comments/12hs2sh/comment/jfqp8vh) in model-based RL because [it uses x100 fewer samples than other algos](https://reddit.com/r/MachineLearning/comments/197jp2b/comment/ki23k8y), and [it can learn to play a wide variety of video games simply from their screen outputs, with no manual tuning or domain knowledge required](https://www.reddit.com/r/MachineLearning/comments/12hs2sh/comment/jfqp8vh), and it's [very recent](https://arxiv.org/pdf/2301.04104.pdf) — this ability to be [trained simply with pixel data from a screen](https://danijar.com/project/dreamerv3) is quite critical because it means it can generalize to any (simple enough) game or task with no need for a simulator like MuZero or other RL algos!
- [Docker](https://docker.com) for the DreamerV3 containerization, so that it can both run on my machine (since `tensorflow-cpu` seems to not be available for Mac M2 chips w/ ARM architecture) and also be easily deployed on my OVH servers (crucial for the future 24/7 livestreams!).
- [Chess.com](https://chess.com)'s "Explore" feature for the pixel-based Chess engine.
- A single [NVIDIA RTX 4090]() GPU to train our Deep RL algorithms. Alternatively, another solid solution would be an [OVH bare metal Linux server](https://eco.ovhcloud.com/en), or even [Vast.ai](https://vast.ai) for on-demand cloud infrastructure.
- Self-hosted GPU+PC server, so that I can have both access to a screen for easy debugging e.g. while using Bluestacks with Clash Royale, and so that I can have it running 24/7 (e.g. for the 'Smallville' LLM) without paying crazy bills to cloud providers.
- [Xvfb](https://en.wikipedia.org/wiki/Xvfb) to create a virtual display where Bluestacks can run in said OVH servers.
- [ffmpeg]() (which btw is what's used in the [Pokemon RL inspiration](https://youtu.be/DcYLT37ImBY?si=ZhejWcVJdS61wrXj&t=1962), incl. switching on/off a headless mode!) to capture the pixels from the virtual display created by Xvfb. We can use ffmpeg's `x11grab` feature to capture the screen content of Xvfb's virtual display as training data.
- Ideally [X](https://help.twitter.com/en/using-x/how-to-use-live-producer#:~:text=Broadcasts%20scheduled%20to%20Start%20later,broadcast%20before%20it%20goes%20live) as a livestream platform for said renders (+ AI stuff to show it is "alive" and "continuously improving on its own"), but perhaps [Twitch](https://twitch.tv) instead since it's the most popular platform for livestreaming and also has a [nice API](https://dev.twitch.tv/docs/api/). -->

## Milestones

![Chess](/img/intro-banner.png)

### Chess

The goal with this project is simple: build a Reinforcement Learning (RL) algo that can play chess better than the top-100 human in the world.

![Chess Training](/img/chess-banner.png)

For initial concept validation we tested on a single Mac M2 CPU (no GPU yet!) for 2 days, and we started occassionally observing emergent behaviour from classic amateur plays and some textbook openings. I have recently ordered an NVIDIA RTX 4090 that I will use to further train the model in single-GPU mode, as the DreamerV3 authors have previously shown that it can learn to play a wide variety of video games (including Minecraft which is remarkably more complex) simply from their screen outputs, with no manual tuning or domain knowledge required — this README will be update with any relevant results once I receive my RTX 4090 and can commit to a ~20 day train run like the authors did for Minecraft's diamond-mining task in their paper.

> _Side note: if you unsure which GPU to use for Machine Learning, [this chart](https://i0.wp.com/timdettmers.com/wp-content/uploads/2023/01/gpu_recommendations.png) is a decent heuristic._

### Clash Royale

TODO: elaborate
<!---(detail HERE the whole pipeline to capture raw pixels from screen, action space, observation space, etc)

Tech Specs:

- For the Action Space we use the 8 possible actions (4 directions, 2 buttons) and the 4 possible cards to play
- For the observation space we use downsampled raw screen pixels, where we currently estimate 448x448x3 being the lower size that allows reading numbers and distinguishing small cards like the skeletons. However this may be a problem, because DreamerV3 uses 64x64x3 for Minecraft and Crafter.-->

|             2x2: Action space too large for training              |              3x3: Balanced training vs. performance               |                 4x4: Unplayable game performance                  |
| :---------------------------------------------------------------: | :---------------------------------------------------------------: | :---------------------------------------------------------------: |
| ![2x2 grid](/img/clash_royale/action-space-analysis-2x2_grid.png) | ![3x3 grid](/img/clash_royale/action-space-analysis-3x3_grid.png) | ![4x4 grid](/img/clash_royale/action-space-analysis-4x4_grid.png) |

## Installation

Install [JAX][jax] and then the other dependencies:

```sh
pip install -r requirements.txt
```

Run a simple training script (if needed, I suggest using `caffeinate` to prevent your machine from sleeping):

```sh
caffeinate python src/main.py
```

There is also the option for a more flexible training script:

```sh
python src/dreamerv3/train.py \
  --logdir ~/logdir/$(date "+%Y%m%d-%H%M%S") \
  --configs crafter --batch_size 16 --run.train_ratio 32
```

### Tips

- All DreamerV3 config options are listed in `configs.yaml` and you can override them
  from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag. Note that multi-GPU support is untested.
- You can run with multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter large`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as TensorBoard summaries. Other outputs like WandB can be
  enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions.
- You can use the `small`, `medium`, `large` config blocks to reduce memory
  requirements. The default is `xlarge`. See the scaling graph above to see how
  this affects performance.
- Many environments are included, some of which require installating additional
  packages. See the installation scripts in `scripts` and the `Dockerfile` for
  reference.
- When running on custom environments, make sure to specify the observation
  keys the agent should be using via `encoder.mlp_keys`, `encode.cnn_keys`,
  `decoder.mlp_keys` and `decoder.cnn_keys`.
- To log metrics from environments without showing them to the agent or storing
  them in the replay buffer, return them as observation keys with `log_` prefix
  and enable logging via the `run.log_keys_...` options.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.

<br/>

---

## About DreamerV3: Mastering Diverse Domains through World Models

[DreamerV3][paper] is a scalable and general reinforcement learning
algorithm that masters a wide range of applications with fixed hyperparameters.

To learn more:

- [Research paper][paper]
- [Project website][website]
- [X (Twitter) summary][tweet]

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](/img/dreamerv3/architecture.png)

DreamerV3 masters a wide range of domains with a fixed set of hyperparameters,
outperforming specialized methods. Removing the need for tuning reduces the
amount of expert knowledge and computational resources needed to apply
reinforcement learning.

![DreamerV3 Benchmark Scores](/img/dreamerv3/performance.png)

Due to its robustness, DreamerV3 shows favorable scaling properties. Notably,
using larger models consistently increases not only its final performance but
also its data-efficiency. Increasing the number of gradient steps further
increases data efficiency.

![DreamerV3 Scaling Behavior](/img/dreamerv3/scores.png)

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104v1.pdf
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441
[example]: https://github.com/danijar/dreamerv3/blob/main/example.py
