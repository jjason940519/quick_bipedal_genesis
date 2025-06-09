# quick_bipedal_genesis
quick bipedal robot training environment
The reinforcement learning genesis enviroment of this repository is based on:
[wheel_legged_genesis]https://github.com/Albusgive/wheel_legged_genesis.git

## System Requirements  
Ubuntu 20.04/22.04/24.04  
python >= 3.10
## Hardware requirements  
NVIDIA/AMD GPU or CPU  
## must（必须）
**Use the main branch of Genesis to install it locally, and you cannot use Genesis 0.2.1 Release**  

## Before running
### 1. Clone repo
run:  
```
git clone https://github.com/jjason940519/quick_bipedal_genesis.git
cd quick_bipedal_genesis
```

### 2. install deps
#### use pdm install
Install pdm, <https://pdm-project.org/en/latest/#installation>, then run
```
pdm install
```

#### or manual install
Install Genesis:  
<https://github.com/Genesis-Embodied-AI/Genesis>  
install tensorboard:    
`pip install tensorboard`  
`pip install pygame`   
`pip install opencv-python`  

install rsl-rl:    
`cd rsl_rl && pip install -e .`  

## Use
### use pdm
test:  
`pdm run locomotion/wheel_legged_eval.py`  
train:  
`pdm run locomotion/wheel_legged_train.py`  

### or manual
test:  
`python locomotion/wheel_legged_eval.py`  
train:  
`python locomotion/wheel_legged_train.py`  

### gamepad & keyboard
**gamepad**  
|key|function|
|---|--------|
|LY|lin_vel|
|RX|ang_vel|
|LT|height_up|
|RT|height_down|
|X|Reset|

**keyboard**
|key|function|
|---|--------|
|W|前进 (Forward)|
|S|后退 (Backward)|
|A|左移 (Move Left)|
|D|右移 (Move Right)|
|Q|左转 (TURN Left)|
|E|右转 (TURN Right)|
|Space|站立 (up)|
|C/Left_Ctrl|下蹲 (down)|
|Shift|静步(Quiet Walking)|
|R|重置环境(Reset)|
## Terrain
You can use the terrain as agent_eval_gym/agent_train_gym/agent_eval_gym/circular  
|terrain|description|
|-------|-----------|
|agent_train_gym|Rough roads and continuous slopes|
|agent_eval_gym|The agent_train_gym of the lite version|
|circular|Circular continuous slopes|

**About custom terrains**
Please refer to the code in `assets/terrain`    

