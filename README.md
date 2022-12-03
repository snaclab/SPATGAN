
Synthetic Traffic Generation with Wasserstein Generative Adversarial Networks(SPATGAN)
===============
This repo provides simple training codes of SPATGAN, simulations, and visualization demo.

The paper is accepted to GLOBECOM 2022. If you find this code is helpful, please cite our paper.

By Chao-Lun Wu, Yu-Ying Chen, Po-Yu Chou, Chih-Yu Wang.

```
@INPROCEEDINGS{spatgan,
  author={Chao-Lun Wu, Yu-Ying Chen, Po-Yu Chou, Chih-Yu Wang},
  booktitle={2022 IEEE Global Communications Conference (GLOBECOM)}, 
  title={Synthetic Traffic Generation with Wasserstein Generative Adversarial Networks},
  year={2022}}
```

Abstract
===============
Network traffic data are critical for network research. With the help of synthetic traffic, researchers can readily generate data for network simulation and performance evaluation.

However, the state-of-the-art traffic generators are either too simple to generate realistic traffic or require the implementation of original applications and user operations.

We propose Synthetic PAcket Traffic Generative Adversarial Networks (SPATGAN) that are capable of generating synthetic traffic. The framework includes a server agent and a client agent, which transmit synthetic packets to each other and take the opponent's synthetic packets as conditional labels for the built-in Timing Synthesis Generative Adversarial Networks (TSynGAN) and a Packet Synthesis Generative Adversarial Networks (PSynGAN) to generate synthetic traffic.

The evaluations demonstrate that the proposed framework can generate traffic whose distribution resembles real traffic distribution.


Requirements
==============
Python 3

TensorFlow 1.15

Detailed requirements are shown in requirements.txt


Running the program
===============

How to train the model?
---------------
execute the following command
```
sh train.sh
```

How to simulate traffic?
---------------
open 2 terminals, execute the following command separately.

server-side:
```
sh simulate_server.sh
```
client-side:
```
sh simulate_client.sh
```

How to use visualization demo?
---------------
execute the following command

traffic flow:
```
sh show_flow.sh
```
pca:
```
sh show_pca.sh
```
tsne:
```
sh show_tsne.sh
```
