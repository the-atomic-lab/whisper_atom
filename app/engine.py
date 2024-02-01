from copy import deepcopy
from io import BytesIO
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import whisperx
import logging
from pyannote.audio import Pipeline as Pyannote
from pydantic import BaseModel
from pydub import AudioSegment
from whisperx.asr import FasterWhisperPipeline
from app.config import EnvVar
import torchaudio

class STT(BaseModel):
    model: FasterWhisperPipeline
    align_model: Optional[Dict[str, tuple]]
    diarizator: Pyannote = None
    stt_infer_batch: int = 16
    device: str

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def load(
        cls,
        model: str,
        align: bool = True,
        diarizator: str = None,
        hg_auth_token: str = None,
        device: str = None,
        compute_type: str = "float16",
        stt_infer_batch: int = 16,
    ):
        """Load a STT pipeline.

        Args:
            model (str): Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large), a path to a converted model directory, or a CTranslate2-converted Whisper model ID from the HF Hub. When a size or a model ID is configured, the converted model is downloaded from the Hugging Face Hub.
            align (bool, optional): Whether to align the output timeline with a wav2vec2 model. Defaults to True.
            diarizator (str, optional): The diarization model to use. Defaults to None.
            hg_auth_token (str, optional): huggingface authorization token. Defaults to None.
            device (str, optional): Device to use for computation ("cpu", "cuda"). Defaults to None.
            compute_type (str, optional): Type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html. Defaults to "float16".
        """
        logging.info("Loading STT model...")
        if ":" in device:
            hardware, device_index = device.split(":")
            device_index = int(device_index)
        else:
            hardware = device
            device_index = 0
        model = whisperx.load_model(
            model, hardware, compute_type=compute_type, device_index=device_index
        )
        if align:
            logging.info("Loading default alignment model...")
            aling_model = {
                "en": cls.load_align_model(
                    lang="en", device=device, model_dir="resources/wav2vec"
                ),
                "zh": cls.load_align_model(
                    lang="zh",
                    device=device,
                    model_name="resources/wav2vec/wav2vec2-large-xlsr-53-chinese-zh-cn",
                ),
            }
        if diarizator is not None:
            logging.info("Loading diarization model...")
            diarizator = Pyannote.from_pretrained(
                diarizator, use_auth_token=hg_auth_token
            )
            diarizator.to(torch.device(device if device is not None else "cpu"))
        return cls(
            model=model,
            align_model=aling_model if align else {},
            diarizator=diarizator,
            device=device,
            stt_infer_batch=stt_infer_batch,
        )

    @staticmethod
    def load_align_model(
        lang: str, model_name: str = None, model_dir: str = None, device: str = None
    ):
        print(f"Loading alignment model {lang} on {device}...")
        return whisperx.load_align_model(
            language_code=lang,
            device=device,
            model_dir=model_dir,
            model_name=model_name,
        )

    def audio2np(self, audio: AudioSegment):
        if audio.frame_rate != 16000:  # 16 kHz
            audio = audio.set_frame_rate(16000)
        if audio.sample_width != 2:  # int16
            audio = audio.set_sample_width(2)
        if audio.channels != 1:  # mono
            audio = audio.set_channels(1)
        arr = np.array(audio.get_array_of_samples())
        arr = arr.astype(np.float32) / 32768.0
        return arr

    def _do_stt(self, audio: AudioSegment, start: int = 0, end: int = -1):
        audio = audio[start:end]
        arr = self.audio2np(audio)
        result = self.model.transcribe(arr, batch_size=self.stt_infer_batch)
        return result

    def _do_align(
        self,
        audio: AudioSegment,
        stt_results: dict,
        start: int = 0,
        end: int = -1,
        lang: str = None,
    ):
        audio = audio[start:end]
        arr = self.audio2np(audio)
        lang = lang if lang is not None else stt_results["language"]
        if lang not in self.align_model:
            self.align_model[lang] = self.load_align_model(lang, self.device)
        model, meta = self.align_model[lang]
        results = whisperx.align(
            stt_results["segments"],
            model,
            meta,
            arr,
            self.device,
            return_char_alignments=False,
        )
        return results

    def _do_diarization(self, audio: AudioSegment, start: int = 0, end: int = -1):
        audio = audio[start:end]
        buffer = BytesIO()
        audio.export(out_f=buffer, format="wav")
        waveform, sample_rate = torchaudio.load(buffer)
        results = self.diarizator({"waveform": waveform, "sample_rate": sample_rate})
        return results

    def apply(
        self,
        audio: AudioSegment,
        enable_align: bool = True,
        enable_diarization: bool = True,
        start: int = 0,
        end: int = -1,
    ):
        logging.info("Start STT...")
        results = self._do_stt(audio, start, end)
        logging.info("STT finished.")

        if enable_align:
            logging.info("Start alignment...")
            results = self._do_align(audio, results, start, end)
            logging.info("Alignment finished.")

        if enable_diarization:
            logging.info("Start diarization...")
            dia_res = self._do_diarization(audio, start, end)
            logging.info("Diarization finished.")
            diarize_df = pd.DataFrame(
                dia_res.itertracks(yield_label=True),
                columns=["segment", "label", "speaker"],
            )
            diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
            diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
            results = whisperx.assign_word_speakers(diarize_df, results)

        return results
