import json
import logging
import os.path
import sys
from pathlib import Path

import click
import torch
import torchaudio
import torchaudio.functional as F
from huggingface_hub import snapshot_download
from natsort import natsorted
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from zonos.conditioning import make_cond_dict
from zonos.model import Zonos

from ._types import Conditioning, Dialog

logger = logging.getLogger(__name__)

OUTPUT_SAMPLING_RATE = 24_000


@click.command()
@click.option(
    "--tts-model",
    type=click.Choice(["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"]),
    default="Zyphra/Zonos-v0.1-transformer",
    help="TTS model to use.",
)
@click.option(
    "--speakers",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory containing speaker embeddings and metadata JSON file.",
)
@click.option(
    "--dialogues",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory containing dialogue JSON files.",
)
@click.option(
    "--inter-turn-msec",
    type=int,
    required=True,
    default=500,
    help="Silence duration between turns in milliseconds.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Output directory.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
)
def main(
    tts_model: str,
    speakers: str,
    dialogues: str,
    inter_turn_msec: int,
    output: str,
    device: str,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        stream=sys.stderr,
    )

    speakers_dir = Path(speakers)
    dialogues_dir = Path(dialogues)

    output_path = Path(output)
    if output_path.exists():
        logger.warning("Output directory already exists.")
    else:
        output_path.mkdir(parents=True)
    output_train_path = output_path / "train"
    output_train_path.mkdir(exist_ok=True)
    metadata_path = output_train_path / "metadata.jsonl"

    snapshot_download(
        repo_id=tts_model,
        repo_type="model",
        revision="main",
    )

    # Load voices
    logger.info("Loading speakers")
    speaker_conds: dict[str, Conditioning] = dict()
    for speaker_file in speakers_dir.glob("*.pt"):
        spekaer_id = os.path.basename(speaker_file).split(".")[0]
        conditioning = Conditioning(
            speaker=torch.load(speaker_file).to(device),
        )
        # Load other conditioning parameters
        cond_metadata_file = speakers_dir / f"{spekaer_id}.json"
        if cond_metadata_file.exists():
            with open(cond_metadata_file) as f:
                metadata = json.loads(f.read())
                if "emotion" in metadata:
                    conditioning.emotion = metadata["emotion"]
                if "speaking_rate" in metadata:
                    conditioning.speaking_rate = metadata["speaking_rate"]
                if "pitch_std" in metadata:
                    conditioning.pitch_std = metadata["pitch_std"]
        speaker_conds[spekaer_id] = conditioning

    # Load Zonos
    logger.info("Loading Zonos")
    tts_model = Zonos.from_pretrained(tts_model).to(device)
    tts_model.eval()

    # Load dialogues
    dialogue_files = natsorted(dialogues_dir.glob("*.json"))
    with logging_redirect_tqdm():
        pbar = tqdm(dialogue_files, desc="Generating audio")
        for dialog_file in pbar:
            pbar.set_description(f"Generating audio ({dialog_file.name})")
            try:
                dialog = Dialog.from_file(dialog_file)
                dialog_id = int(dialog.dialogue_id)
                turn_wavs = []
                texts = []
                for turn_id, utterance in enumerate(dialog.utterances):
                    text = utterance.text
                    speaker_id = utterance.interlocutor_id
                    cond = speaker_conds.get(utterance.interlocutor_id, None)
                    if cond is None:
                        logging.warning("No speaker found for %s", speaker_id)
                        continue

                    sampling_params = cond.sampling_params()
                    cond.speaker = torch.load("./speakers/zonos/base/female/004.pt").to(device)

                    texts.append(text)
                    cond_dict = make_cond_dict(text, **cond.conditioning_params())
                    conditioning = tts_model.prepare_conditioning(cond_dict).to(device)
                    codes = tts_model.generate(conditioning, sampling_params=sampling_params, progress_bar=False)
                    wavs = tts_model.autoencoder.decode(codes).to("cpu")

                    resampled_wav = F.resample(wavs[0], tts_model.autoencoder.sampling_rate, OUTPUT_SAMPLING_RATE)
                    turn_wavs.append(resampled_wav)
                    logger.debug(f"Generated {dialog_id:05d}_{turn_id:03d}")

                # Create contiguous audio for the dialogue
                audio_file = output_train_path / f"{dialog_id:05d}.flac"
                contiguous_wavs = []
                for i in range(len(turn_wavs)):
                    silence = torch.zeros((OUTPUT_SAMPLING_RATE * inter_turn_msec) // 1000).unsqueeze(0)
                    contiguous_wavs.append(turn_wavs[i])
                    if i < len(turn_wavs) - 1:
                        contiguous_wavs.append(silence)

                congiguous_wav = torch.cat(contiguous_wavs, dim=-1)
                torchaudio.save(audio_file, congiguous_wav, OUTPUT_SAMPLING_RATE)

                # Save metadata
                with open(str(metadata_path), "a+") as f:
                    metadata = {
                        "file_name": os.path.basename(audio_file),
                        "text": "".join(texts),
                        "speaker_id": speaker_id,
                    }
                    f.write(json.dumps(metadata, ensure_ascii=False))
                    f.write("\n")

            except Exception as e:
                logger.exception(f"Failed on {dialog_file}", exc_info=e)
                break


if __name__ == "__main__":
    main()
