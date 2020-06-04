

# Direct Policy Gradients - RL with A* sampling and direct optimization

Pytorch impelentation for MiniGrid and DeepSea experiments from the paper "Direct Policy Gradients: Direct Optimization of Policies in Discrete Action Spaces"

## Installation

clone repository, create new virtualenv and install dependencies:
```bash
git clone https://github.com/GuyLor/reinforcement_learning.git
python3 -m venv direct_rl
source direct_rl/bin/activate
cd reinforcment_learning
pip3 install -r requirements.txt
```

## Usage

train from scratch:
```bash
python run.py --train
```  

let the trained policy to "play" after training:
  
```bash
python run.py --train --play 
```  
save and/or load the model after training:
```bash
  python run.py --train --play --save_path my_policy_model_new.pkl --load_path my_policy_model.pkl
```  
open tensorboard:
```bash
  tensorboard --logdir logs
```  

![multi-room](https://user-images.githubusercontent.com/18574572/83604692-4711cd00-a57f-11ea-916b-c208834c2dea.gif)

## License
[MIT](https://choosealicense.com/licenses/mit/)
