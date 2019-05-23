# Direct Policy Gradients - RL with A* sampling and direct optimization

python3 -m venv direct_rl

source direct_rl/bin/activate

pip3 install -r requirements.txt

./direct_rl/bin/jupyter notebook

-------------------------------------
to train from scratch:
  python run.py --train

to play with a trained model after training:
  python run.py --train --play 

save and/or load the model after training:
  python run.py --train --play --save_path my_policy_model.pkl --load_path my_policy_model.pkl


