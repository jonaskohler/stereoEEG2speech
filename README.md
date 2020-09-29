# stereoEEG2speech
We provide code for a seq2seq architecture with Bahdanau attention designed to map stereotactic EEG data from human brains to spectrograms, using the PyTorch Lightning frameworks. The regressed spectograms can then be used to synthesize actual speech (for example) via the flow based generative Waveglow architecture.


## Data
Stereotactic electroencephalogaphy (sEEG) utilizes localized, penetrating depth electrodes to measure electrophysiological brain activity. The implanted electrodes generally provide a sparse sampling of a unique set of brain regions including deeper brain structures such as hippocampus, amygdala and insula that cannot be captured by superficial measurement modalities such as electrocorticography (ECoG). As a result, sEEG data provides a promising bases for future research on Brain Computer Interfaces (BCIs) [1].

In this project we use sEEG data from patients with 8 sEEG electrode shafts of which each shaft contains 8-18 contacts. Patients read out sequences of either words or sentences over a duration of 10-30 minutes. Audio is recorded at 44khz and EEG data is recoded at 1khz. As an intermediate representation, we embed the audio data in mel-scale spectrograms of 80 bins.


## Network architecture

Existing models in speech synthesis from neural activity in the human brain rely mainly on fully connected and convolutional models (e.g. [2]). Yet, due to the clear temporal structure of this task we here propose the use of RNN based architectures.

![Network architecture](/model_overview.png)


### EEG to Spectograms

In particular, we provide code for an RNN that presents an adaption NVIDIAs Tacotron2 model [3] to the specific type of data at hand. As such, the model consists of an encoder-decoder architecture with an upstream CNN that allows to downsample and filter the raw EEG input. 

(i) CNN: We present data of 112 channels to the network in a sliding window of 200ms with a hop of 15ms at  1024Hz. At first, a three layer convnet parses and downsamples this data about 100Hz and reduces the number of channels to 75. The convolution can be done one or two dimensional.

(ii) RNN: We add sinusoidal positional embeddings (32) to this sequence and feed it into a bi-directional RNN encoder with 3 layers of GRUs which embeds the data in a hidden state of 256 dimensions. Furthermore, we employ a Bahdanau attention layer on the last layer activations of the encoder.  

(iii) Prediction:
Both results are passed into a one layer GRU decoder which outputs a 256 dimensional representation for each point in time. A fully connected ELU layer followed by a linear layer regresses spectrogram predictions in 80 mel bins. On the one hand, this prediction is passed trough a fully connected Prenet which re-feeds the result into the GRU decoder for the next time step. On the other hand, it is also passed through a five layer 1 d convolutional network. The output is concatenated with the original prediction to give the final spectrogram prediction.

The default loss in our setting is MSE, albeit we also offer a cross entropy based loss calculation in the case of discretized mel bins (e.g. arising from clustering) which can make the task easier for smaller datasets. Moreover, as sEEG electrodes placement usually varies across patients, the model presented here is to be trained on each patient individually. Yet, we also provide code for joint training with a contrastive loss that incentives the model
to minimize the embedding distance within but maximize across patients.
 

### Spectograms to audio
The predicted spectrograms can be passed trough any of the state of the art generative models for speech synthesis from spectograms. The current code is designed to create mel spectograms that can be fed right away into the flow based generative WaveGlow model from NVIDIA [4].


[1] Herff, Christian, Dean J. Krusienski, and Pieter Kubben. "The Potential of Stereotactic-EEG for Brain-Computer Interfaces: Current Progress and Future Directions." Frontiers in Neuroscience 14 (2020): 123.

[2] Angrick, Miguel, et al. "Speech synthesis from ECoG using densely connected 3D convolutional neural networks." Journal of neural engineering 16.3 (2019): 036019.

[3] Shen, Jonathan, et al. "Natural tts synthesis by conditioning wavenet on mel spectrogram predictions." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.

[4] Prenger, Ryan, Rafael Valle, and Bryan Catanzaro. "Waveglow: A flow-based generative network for speech synthesis." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019. 
