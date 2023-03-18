import matplotlib.pyplot as plt
import numpy as np

from midigpt import GPT
from midigpt.datasets import TextCharacterTokenizer

loss_history = np.genfromtxt("loss_history.txt")

plt.style.use("dark_background")
plt.plot(loss_history)
plt.ylabel("Log Loss", fontsize=15)
plt.xlabel("Iteration", fontsize=15)
plt.savefig("loss_history.png", dpi=200)

tokenizer = TextCharacterTokenizer.from_file("tiny-shakespeare.txt")

model = GPT.from_checkpoint("best_model.ckpt")

context = "KING JOHNNY:\nTo be, or not to be: that is the question."

generation = model.generate(tokenizer.encode(context), 5000, do_sample=True)

with open("generated-shakespeare.txt", "w") as f:
    f.write(tokenizer.decode(generation.flatten().tolist()))
