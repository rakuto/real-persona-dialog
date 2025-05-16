import os
import typing as tp

import torch
from pydantic import BaseModel, ConfigDict, Field


class Utterance(BaseModel):
    utterance_id: int
    interlocutor_id: str
    text: str
    timestamp: str


class Dialog(BaseModel):
    dialogue_id: int
    interlocutors: list[str]
    utterances: list[Utterance]

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "Dialog":
        with open(path) as f:
            return Dialog.model_validate_json(f.read())


class Conditioning(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    language: str = Field("ja")
    speaker: torch.Tensor
    emotion: list[float] | None = None
    pitch_std: float | None = None
    speaking_rate: float | None = None

    def conditioning_params(self) -> dict[str, tp.Any]:
        params = dict(speaker=self.speaker, language=self.language)
        if self.emotion is not None:
            params["emotion"] = self.emotion
        if self.pitch_std is not None:
            params["pitch_std"] = self.pitch_std
        if self.speaking_rate is not None:
            params["speaking_rate"] = self.speaking_rate

        return params

    def sampling_params(self) -> dict[str, tp.Any]:
        samplers = dict(temperature=1.0, top_p=0.9, min_p=0.1)

        return samplers
