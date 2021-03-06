# DissectOfflineRL

Dissect Offline Reinforcement Learning, what do we need wrt. datasets and buffer strategies to succeed in this setting.

### Current State:

#### Algorithms

Algorithms in this domain need to be off-policy, the first 4 being simple value
learning algorithms and the fifth is an actor-critic approach. The remaining three aim towards safe q-iteration. Behavioral Cloning serves
as a baseline.

- [x] [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) ([Double](https://arxiv.org/abs/1509.06461))
- [x] [REM](https://arxiv.org/abs/1907.04543)
- [x] [BCQ](https://arxiv.org/abs/1910.01708)
- [x] [QR-DQN](https://arxiv.org/abs/1710.10044)
- [x] [SAC](https://arxiv.org/abs/1801.01290) ([Discrete](https://arxiv.org/abs/1910.07207))
- [ ] [SPIBB](https://arxiv.org/abs/1712.06924)
- [ ] [CQL](https://arxiv.org/abs/2006.04779)?? many report problems to implement
- [ ] [QRr / BRr](https://offline-rl-neurips.github.io/program/offrl_41.html)
- [x] Behavioral Cloning (BC)

#### Environments

The following environments should be of progressing difficulty, thus
the classic control problems being relatively simple to solve (except MountainCar).
Minigrid has more high difficult settings, being kind of in the medium range.
MinAtar is kind of as challenging as the original Atari games with similar dynamics, 
but has much less computational overhead. All environment can be solved with Linear Networks
and I will not try to do otherwise. This has the implication that the regularisation
effects of CNN's will be lost.

- [x] [Classic control](https://gym.openai.com/envs/#classic_control)
    - [x] [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/)
    - [x] [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/)
    - [ ] [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/)
    - [ ] [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)
- [x] [Minigrid](https://github.com/maximecb/gym-minigrid)
    - [x] [MiniGrid-Empty-Random-6x6-v0](https://github.com/maximecb/gym-minigrid#empty-environment)
    - [x] [MiniGrid-Unlock-v0](https://github.com/maximecb/gym-minigrid#unlock-environment)
    - [x] [MiniGrid-DistShift1-v0](https://github.com/maximecb/gym-minigrid#distributional-shift-environment)
      - there exists a MiniGrid-DistShift2-v0 with a slight modification, can test distribution shift on that
    - [x] [MiniGrid-SimpleCrossingS9N1-v0](https://github.com/maximecb/gym-minigrid#simple-crossing-environment)    
    - [x] [MiniGrid-LavaCrossingS9N1-v0](https://github.com/maximecb/gym-minigrid#lava-crossing-environment)
    - [ ] TBD
- [ ] [MinAtar](https://github.com/kenjyoung/MinAtar) ?
    - [ ] Breakout
    - [ ] Seaquest
    - [ ] Asterix
    - [ ] Freeway
    - [ ] Space Invaders 
    - [gym registration](https://github.com/qlan3/gym-games)
    
#### Experimental ideas

  
  - [ ] Follow the line of [Kmec et al., 2020](https://arxiv.org/abs/2011.14379) to assess different dataset
    generation strategies on different environments by different strategies/policies (expert, ER, medium(random + expert / high stochastic expert), expert 
  - [ ] Create metrics to assess collected offline datasets (e.g. performance of BC and entropy, reward / episode structure, 
    diversity)
  - [ ] Categorization scheme
    - [ ] Characterize RLUnplugged / D4RL
  - [ ] Benchmark various off-policy algorithms on the assessed problems, give recommendations of algorithms depending on problem structure
and dataset metrics.
    

  - [ ] ??? If possible, assess offline performance vs online performance within a certain budget
    - [ ] Observed SAC fail in the grid world envs trained online, but heavily exceed when trained on ER buffer of DQN.
  - [ ] ??? Implement improvement to SPIBB (Laroche et al., 2019)(https://arxiv.org/pdf/1712.06924.pdf) which is a nice idea but not generally applicable yet as it uses density to assess whether state/action pair is part of the
dataset, I would like to try an overfitted autoencoder to answer the question (i.e. train several epochs, measure worst recreation error and if new state/action
    is within this bound it is in the dataset). (GO-explore!, #exploration)

---
    
  - [ ] Get insights into the benefit of the Experience Replay Buffer
    - seed the environment before every episode rollout both for training and testing
  - [ ] Try to train same algorithm as behavioral on ER buffer, not on full dataset but with progression.
  - [ ] Train offline algorithm on random data first, then switch to high reward dataset. Curriculum!
  - [ ] What happens if we train DQN with different network seed in exactly the same fashion as behavioral?
