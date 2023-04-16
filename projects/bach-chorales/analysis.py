from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from midigpt import GPT, TetradPlayer
from midigpt.datasets import BachChoralesEncoder


def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]


loss_history = np.genfromtxt("loss_history.txt")
plt.style.use("dark_background")
plt.plot(loss_history)
plt.ylabel("Log Loss", fontsize=15)
plt.xlabel("Iteration", fontsize=15)
plt.savefig("loss_history.png", dpi=200)

jsb_chorales_dir = Path("jsb_chorales")
test_files = sorted(jsb_chorales_dir.glob("test/chorale_*.csv"))
test_chorales = load_chorales(test_files)

model = GPT.from_checkpoint("best_model.ckpt", map_location=torch.device("cpu"))
encoder = BachChoralesEncoder()
seed_notes = encoder.encode(torch.as_tensor(test_chorales[10][:8]).flatten())

for temperature in [0.5, 1.5, 1.8]:
    chords = encoder.decode(model.generate(seed_notes, 120, do_sample=True, temperature=temperature))
    TetradPlayer().to_wav(chords.flatten().reshape(-1, 4).cpu(), f"generated-bach-{temperature}.wav", tempo=220)
