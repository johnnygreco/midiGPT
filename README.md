# 🎹 midiGPT

Inspired by Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT), I am building `midiGPT` as a learning-in-public (LIP) project. The goal is to build a library that makes it easy to train a decoder-only transformer model (similar to GPT) on MIDI data. 

<br>
<br>

# 🚧 Under Construction 🚧

<br>
<br>
<br>
<br>
<br>

### 📝 Notes
To use pyfluidsynth with pretty_midi, I had to brew install fluidsynth and create a symlink to the library:

```bash
brew install fluidsynth

ln -s /opt/homebrew/opt/fluidsynth/lib/libfluidsynth.dylib /opt/homebrew/Caskroom/miniforge/base/envs/midiGPT/lib/libfluidsynth.dylib
```