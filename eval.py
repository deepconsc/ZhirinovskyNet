import argparse
import numpy as np 

from modules.model import ZhirinovskyNet
from modules.preprocessor import Sampler


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", type=str, required=False)
parser.add_argument("--steps", default=1, type=int, required=False)
args = parser.parse_args()

model = ZhirinovskyNet(device=args.device)
sampler = Sampler()

Trials = []

for idx in range(args.steps):
    input_tensor = sampler.sample()
    prob = model.forward(input_tensor)
    Trials.append(prob)

alive = np.mean(Trials) > 0.5 

print(f"{args.steps} Trial Runs have been finished.")

if not alive: 
    print(f"Congratulations! Theoretically he turns out to be DEAD.")
if alive: 
    print(f"We're sorry to inform you that theoretically he's alive. We hope that practical, real-world results are better.")