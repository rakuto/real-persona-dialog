# RealPersonaChat Speech Dataset

This repository contains a script to generate audio conversation dataset
from [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat/) dataset using
[Zonos](https://github.com/Zyphra/Zonos) TTS engine.

## Create Speaker Conditioning

Speaker conditionings are created from personality and persona given in RealPersonaChat dataset by using LLM to create
conditioning vectors and paraemters. There are male and female base voices are randomly sampled according to gender
given in the dataset.

## Generate conversation audio

Running on GPU cloud by using [Skypilot](https://docs.skypilot.co/en/latest/) is a quick way to generate audio files.

```shell
sky launch -c persona-chat run_on_lambda.yaml
```

Audios are generated in `outputs` folder following Hugging
Face [Audio Dataset](https://huggingface.co/docs/hub/en/datasets-audio)
format.
