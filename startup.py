import os

# create necessary folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# launch tensorboard
os.system('tensorboard --logdir=runs')