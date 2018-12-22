# reinforcement_learning - Astar sampling and direct optimization

python3 -m venv direct_rl

source direct_rl/bin/activate

pip3 install -r requirements.txt

./direct_rl/bin/jupyter notebook

-------------------------------------
to train from scratch:
  python run.py --train

to play with a trained model:
  python run.py --play --load_path policy_state_dicts_multiroom.pkl

