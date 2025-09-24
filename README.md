🐡: Welcome to PreferGrow, this is a implementation of ***Fading to Grow: Growing Preference Ratios via Preference Fading Discrete Diffusion for Recommendation***

## :zero:  ​ Datasets

https://drive.google.com/file/d/1ri0MRUScdA0AxZiAxj8N21QOE1rc0Nh8/view?usp=sharing

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
