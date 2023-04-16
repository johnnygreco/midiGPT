# 🎼 Bach Chorales

The Bach Chorales dataset consists of 382 chorales composed by Johann Sebastian Bach. Each chorale is 100 to 640 time steps long, where each time step contains 4 integers that correspond to a note's index on a piano (except for the value 0, which corresponds to a rest).

I trained a GPT model to generate new Bach Chorales by predicting the next note in sequences of arpeggios. The model had 24 self-attention blocks, each consisting of 8 attention heads and an embedding size of 64. total number of parameters was 622,383.

## Loss History
<img src="loss_history.png" width=80% height=80%>

## Example Generations

Here are some example generations that are seeded with two chords from a random Bach Chorale from the test set. 

- ♫ [temperature = 0.5](generated-bach-0.5.wav)

- ♬ [temperature = 1.5](generated-bach-1.5.wav)

- 🎶 [temperature = 1.8](generated-bach-1.8.wav)
