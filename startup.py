import os

# create necessary folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# launch tensorboard, can change launch folder if necessary
os.system('tensorboard --logdir=runs')