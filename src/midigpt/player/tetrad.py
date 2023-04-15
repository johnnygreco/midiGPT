"""
Tools for playing tetrad note music in the format of the midi Bach chorales.

Adapted from Aurélien Geron's solution discussed here https://github.com/ageron/handson-ml2/issues/82.

Here's his description of the approach:

To play these notes, the simplest option is probably to generate a MIDI file, I believe there are some 
easy-to-use libraries for this, but that's not what I did. Just for fun I decided to synthesize the sound 
waves directly. :) The standard pitch A on the 4th octave is 440Hz, so if you generate a sound wave with 
that frequency, you will hear that note. It corresponds to note number 69 in the file (the notes start at 
C on "octave -1", then there are 12 semi-tones per octave, so C0 is number 12, C1 is number 24 and so on 
up to C4 which is number 60, and A4 which is number 69). Let's call r the frequency ratio between one note 
and the next (going up one semi-tone). Since the frequency doubles when you go up one octave, then r^12 
must be equal to 2. So r=2^(1/12). All of this gives us the equation to find the frequency of a note given 
its index: frequency = 440 * 2**((note - 69)/12). 
"""

import numpy as np
from IPython.display import Audio, display
from scipy.io import wavfile

__all__ = ["TetradPlayer"]


class TetradPlayer:
    def __init__(self, sample_rate=44100, tempo=120, amplitude=0.1):
        self.sample_rate = sample_rate
        self.tempo = tempo
        self.amplitude = amplitude

    @staticmethod
    def notes_to_frequencies(notes):
        # Frequency doubles when you go up one octave; there are 12 semi-tones
        # per octave; Note A on octave 4 is 440 Hz, and it is note number 69.
        return 2 ** ((np.array(notes) - 69) / 12) * 440

    @staticmethod
    def frequencies_to_samples(frequencies, tempo, sample_rate):
        note_duration = 60 / tempo  # the tempo is measured in beats per minutes
        # To reduce click sound at every beat, we round the frequencies to try to
        # get the samples close to zero at the end of each note.
        frequencies = np.round(note_duration * frequencies) / note_duration
        n_samples = int(note_duration * sample_rate)
        time = np.linspace(0, note_duration, n_samples)
        sine_waves = np.sin(2 * np.pi * frequencies.reshape(-1, 1) * time)
        # Removing all notes with frequencies ≤ 9 Hz (includes note 0 = silence)
        sine_waves *= (frequencies > 9.0).reshape(-1, 1)
        return sine_waves.reshape(-1)

    @classmethod
    def chords_to_samples(cls, chords, tempo, sample_rate, amplitude=1.0):
        freqs = cls.notes_to_frequencies(chords)
        freqs = np.r_[freqs, freqs[-1:]]  # make last note a bit longer
        merged = np.mean([cls.frequencies_to_samples(melody, tempo, sample_rate) for melody in freqs.T], axis=0)
        n_fade_out_samples = sample_rate * 60 // tempo  # fade out last note
        fade_out = np.linspace(1.0, 0.0, n_fade_out_samples) ** 2
        merged[-n_fade_out_samples:] *= fade_out
        return amplitude * merged

    def _get_sampling_kwargs(self, **kwargs):
        return dict(
            amplitude=kwargs.get("amplitude", self.amplitude),
            sample_rate=kwargs.get("sample_rate", self.sample_rate),
            tempo=kwargs.get("tempo", self.tempo),
        )

    def play_in_notebook(self, chords, **kwargs):
        display(self.to_audio(chords, **kwargs))

    def to_audio(self, chords, **kwargs):
        samples = self.chords_to_samples(chords, **self._get_sampling_kwargs(**kwargs))
        return Audio(samples, rate=kwargs.get("sample_rate", self.sample_rate))

    def to_wav(self, chords, file_path, **kwargs):
        samples = self.chords_to_samples(chords, **self._get_sampling_kwargs(**kwargs))
        samples = (2**15 * samples).astype(np.int16)
        wavfile.write(file_path, kwargs.get("sample_rate", self.sample_rate), samples)
