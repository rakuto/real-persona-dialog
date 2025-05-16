"""
Generate speaker conditioning for Zonos TTS.

Usage: uv run python create_speaker_conditioning.py --input /path/to/interlocutors.json --output speakers/zonos
"""

import json
import os
import random
import re
import shutil
from pathlib import Path

import click
from google import genai
from tqdm import tqdm

NUM_VOICES = {
    "female": 5,
    "male": 2,
}


@click.command()
@click.option(
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to interlocutors.json file",
)
@click.option("--output", type=click.Path(), required=True)
@click.option("--model", type=str, default="gemini-2.0-flash")
def main(input: str, output: str, model: str):
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("GOOGLE_API_KEY is not set")
    client = genai.Client(api_key=google_api_key)

    output_dir = Path(output)
    if not output_dir.exists():
        output_dir.mkdir()

    with open(input) as f:
        speakers = json.load(f)

    for speaker_id, characteristic in tqdm(speakers.items(), desc="Generating conditioning"):
        personality = {
            "persona": characteristic["persona"],
            "personality": characteristic["personality"],
            "demographic_information": characteristic["demographic_information"],
        }
        gender = characteristic["demographic_information"]["gender"].lower()
        if gender not in ["female", "female"]:
            gender = "male"

        voice = f"speakers/zonos/base/{gender}/{random.randint(1, NUM_VOICES[gender]):03d}.pt"
        shutil.copy(voice, output_dir / f"{speaker_id}.pt")

        response = client.models.generate_content(
            model=model,
            contents=[
                f"""\
You will be given persona and personality of a speaker and description of TTS paraemters. \
You the generate TTS parameters for the speaker matching th given persona and personality. \
Be sure to only return parameters in valid JSON format.

Personality:
{json.dumps(personality)}

TTS parameters:
---
# Emotion vector from 0.0 to 1.0
#   Is entangled with pitch_std because more emotion => more pitch variation
#                     VQScore and DNSMOS because they favor neutral speech
#
#                       Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
emotion: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],

# Standard deviation for pitch (0 to 400), should be 
#   20-45 for normal speech, 
#   60-150 for expressive speech, 
#   higher values => crazier samples
pitch_std: float = 20.0,

# Speaking rate in phonemes per minute (0 to 40). 16 is normal. 30 is very fast, 10 is slow.
speaking_rate: float = 16.0,
---

Geneate TTS parameters for the speaker:

{{
  "emotion": <emotion vector>,
  "pitch_std": <pitch_std>,
  "speaking_rate": <speaking_rate>,
}}\
"""
            ],
        )

        output_path = output_dir / f"{speaker_id}.json"
        m = re.match(r"```json\s+(.*)\s+```", response.text, re.M | re.DOTALL)
        if m:
            condition = m[1]
        else:
            condition = response.text

        with open(str(output_path), "w") as f:
            f.write(condition)


if __name__ == "__main__":
    main()
