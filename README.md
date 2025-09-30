<div align=center>

<h1>Fading to Grow: Growing Preference Ratios via Preference Fading Discrete Diffusion for Recommendation</h1>

<img src="https://img.shields.io/badge/License-MIT-blue" alt="license">

<div>
      <a href="https://Hugo-Chinn.github.io//" target="_blank">Guoqing Hu</a><sup>1</sup>,
      <a href="https://anzhang314.github.io/" target="_blank">An Zhang</a><sup>1&#8224</sup>,
      Shuchang Liu<sup>2</sup>,
      <a href="https://github.com/maowenyu-11" target="_blank">Wenyu Mao</a><sup>1</sup>,
      <a href="https://wujcan.github.io/" target="_blank">Jiancan Wu<sup>1</sup>,
      <a href="https://ftttank.github.io/author/xun-yang/" target="_blank">Xun Yang</a><sup>1</sup>,
      Xiang Li<sup>2</sup>,
      Lantao Hu<sup>2</sup>,
      Kun Gai<sup>2</sup>,
      <a href="https://xiangwang1223.github.io./" target="_blank">Xiang Wang</a><sup>1</sup>,

<div>
  <sup>1</sup>University of Science and Technology of China, <sup>2</sup>Independent Researcher
       </div>   
<div>
<sup>+</sup> Corresponding author. 
   </div>

</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13-EE4C2C.svg)](https://pytorch.org/get-started/previous-versions/)
[![CUDA 11.7](https://img.shields.io/badge/CUDA-11.7-76B900.svg)](https://developer.nvidia.com/cuda-11-7-0-download-archive)
[![cuDNN 8.5](https://img.shields.io/badge/cuDNN-8.5-76B900.svg)](https://developer.nvidia.com/rdp/cudnn-archive)
[![OS Linux](https://img.shields.io/badge/OS-Linux-informational.svg)](https://www.kernel.org/)


</div>

## LRMs plan reasoning strengths with pre-allocated direction vectors

![illustration](teaser_github.png)

In our paper, we reveal that:

- LRMs plan reasoning strengths in advance, even before the generation of the first reasoning token.
- As the difficulty of questions increases, the reasoning strength of LRMs also increases. Meanwhile, the activations of different difficulty levels shift towards the same direction, with the magnitude of this direction controlling the reasoning strength.
- Steering with the pre-allocated direction vectors can change the reasoning strength of LRMs, which further impacts the final performance.

## Usage

🐡: Welcome to PreferGrow, this is a implementation of ***Fading to Grow: Growing Preference Ratios via Preference Fading Discrete Diffusion for Recommendation***


Generate the main results (i.e., linear regression, direction vector extraction and analysis) in this paper:

```bash
bash scripts/analyze.sh
```

Evaluate the effect of steering:

```bash
python AnalyzeSteerFull.py
```

Replicate the results of the overthink detection before generation:

```bash
python eval_overthink.py
```

Replicate the results of the efficient inference:

```bash
python efficient_reasoning.py
```

## Activation Steering with vLLM

We implement the activation steering with vLLM, which can be found in `steer_qwen2_vllm.py` file.
To enable the usage of vLLM, you need to set the environment variable `steering_vector_path` to the path of the steering vector.

```bash
export steering_vector_path=/path/to/steering_vector.npy
```


## ☎️ Contact

Please contact the first author for any questions.

- Leheng Sheng, leheng.sheng@u.nus.edu

## 🌟 Citation

If you find our work useful, please kindly consider citing our work as follows:

```bibtex
@article{sheng2025reasoning,
  title={On Reasoning Strength Planning in Large Reasoning Models},
  author={Sheng, Leheng and Zhang, An and Wu, Zijian and Zhao, Weixiang and Shen, Changshuo and Zhang, Yi and Wang, Xiang and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2506.08390},
  year={2025}
}
```

## :zero:  ​ Datasets

[https://drive.google.com/file/d/1ri0MRUScdA0AxZiAxj8N21QOE1rc0Nh8/view?usp=sharing](https://drive.google.com/file/d/1flFK3TPTiLm6e7WzijVu9gwSi6pD526g/view?usp=drive_link)

https://drive.google.com/file/d/1fsQUa92UV9_MqcGKqDhK9Sh4JrlbblLZ/view?usp=drive_link

## :one:  ​ Guide for Running PreferGrow

### :walking_man: Hybrid Settings

```sh
nohup python -u single_train.py cuda=0 random_seed=100 training.data="Steam" graph.type="hybrid" graph.gamma=0.99999 graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.1 optim.lr=0.001 model.score_flag=False loss_type="score_entropy" model.score_flag=True model.score_method="oricos" > ./log/Steam/RS2_ABest_PreferGrow_HybridW0.99999_dim256_lr1e-3_p0.1_SE_oricos 2>&1 &
nohup python -u single_train.py cuda=0 random_seed=100 training.data="ML1M" graph.type="hybrid" graph.gamma=0.9999 graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.1 optim.lr=0.0001 model.score_flag=False loss_type="score_entropy" model.score_flag=True model.score_method="oricos" > ./log/ML1M/RS2_ABest_PreferGrow_HybridW0.9999_dim256_lr1e-4_p0.1_SE_oricos 2>&1 &
nohup python -u single_train.py cuda=4 random_seed=100 training.data="Beauty" graph.type="hybrid" graph.gamma=0.999 graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.1 optim.lr=0.0001 model.score_flag=False loss_type="score_entropy" model.score_flag=True model.score_method="oricos" > ./log/Beauty/RS2_ABest_PreferGrow_HybridW0.999_dim256_lr1e-4_p0.1_SE_oricos 2>&1 &
nohup python -u single_train.py cuda=4 random_seed=100 training.data="ATG" graph.type="hybrid" graph.gamma=0.9999 graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.2 optim.lr=0.001 model.score_flag=False loss_type="score_entropy" model.score_flag=True model.score_method="oricos" > ./log/ATG/RS2_ABest_PreferGrow_HybridW0.9999_dim256_lr1e-3_p0.2_SE_oricos 2>&1 &
nohup python -u single_train.py cuda=5 random_seed=100 training.data="ASO" graph.type="hybrid" graph.gamma=0.9999 graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.2 optim.lr=0.001 model.score_flag=False loss_type="score_entropy" model.score_flag=True model.score_method="oricos" > ./log/ASO/RS2_ABest_PreferGrow_HybridW0.9999_dim256_lr1e-3_p0.2_SE_oricos 2>&1 &
```
### :walking_man: Adaptive Settings
```sh
nohup python -u single_train.py cuda=6 random_seed=100 training.data="ML1M" graph.type="adaptive" graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.2 optim.lr=0.0001 model.score_flag=False loss_type="score_entropy" model.score_flag=False model.score_method="oricos" > ./log/ML1M/UserProbs_PreferGrow_Adaptive+1_dim256_lr1e-4_p0.2_SE_oricos 2>&1 &
nohup python -u single_train.py cuda=6 random_seed=100 training.data="Steam" graph.type="adaptive" graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.05 optim.lr=0.001 model.score_flag=False loss_type="score_entropy" model.score_flag=False model.score_method="oricos" > ./log/Steam/UserProbs_PreferGrow_Adaptive+1_dim256_lr1e-3_p0.05_SE_oricos 2>&1 &
nohup python -u single_train.py cuda=3 random_seed=100 training.data="Beauty" graph.type="adaptive" graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.1 optim.lr=0.0001 model.score_flag=False loss_type="score_entropy" model.score_flag=False model.score_method="oricos" > ./log/Beauty/UserProbs_PreferGrow_Adaptive+1_dim256_lr1e-4_p0.1_SE_oricos 2>&1 &
nohup python -u single_train.py cuda=4 random_seed=100 training.data="ATG" graph.type="adaptive" graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.2 optim.lr=0.0001 model.score_flag=False loss_type="score_entropy" model.score_flag=False model.score_method="oricos" > ./log/ATG/UserProbs_PreferGrow_Adaptive+1_dim256_lr1e-4_p0.2_SE_oricos 2>&1 &
nohup python -u single_train.py cuda=5 random_seed=100 training.data="ASO" graph.type="adaptive" graph.is_disliked_item=True model.hidden_size=256 model.cond_dim=256 training.nonpreference_user_ratio=0.2 optim.lr=0.0001 model.score_flag=False loss_type="score_entropy" model.score_flag=False model.score_method="oricos" > ./log/ASO/UserProbs_PreferGrow_Adaptive+1_dim256_lr1e-4_p0.2_SE_oricos 2>&1 &
```
