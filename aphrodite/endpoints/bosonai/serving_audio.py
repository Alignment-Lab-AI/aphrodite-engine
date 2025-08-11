# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
import os
import time
import traceback
from collections.abc import AsyncGenerator, AsyncIterator
from functools import lru_cache
from typing import Any, Final, Optional

import librosa
import numpy as np
from fastapi import Request
from pydub import AudioSegment
from starlette.datastructures import State

from aphrodite.common.config import ModelConfig
from aphrodite.engine.protocol import EngineClient
from aphrodite.endpoints.chat_utils import (ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         parse_chat_messages_futures,
                                         resolve_chat_template_content_format)
from aphrodite.endpoints.openai.protocol import (AudioSpeechRequest,
                                              ChatCompletionMessageParam,
                                              RequestResponseMetadata)
from aphrodite.endpoints.openai.serving_engine import OpenAIServing
from aphrodite.endpoints.openai.serving_models import OpenAIServingModels
from aphrodite.inputs import TokensPrompt
from aphrodite.common.logger import setup_logger
from aphrodite.modeling.models.higgs_audio_tokenizer import AudioTokenizer
from aphrodite.common.outputs import RequestOutput
from aphrodite.transformers_utils.tokenizer import AnyTokenizer

from .utils import create_audio_chunk


OPENAI_TTS_SAMPLING_RATE = 24000
OPENAI_TTS_BIT_DEPTH = 16
OPENAI_TTS_CHANNELS = 1

TTS_SYSTEM_PROMPT = "Convert text to speech with the same voice."


@lru_cache(maxsize=50)
def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the MP3 file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def pcm_to_target_format_bytes(pcm_data: np.ndarray, response_format: str,
                               original_sr: int, target_sr: int):
    audio_pcm16 = (pcm_data * np.iinfo(np.int16).max)\
                    .clip(np.iinfo(np.int16).min, np.iinfo(np.int16).max)\
                    .astype(np.int16)
    if response_format == "pcm":
        return audio_pcm16.tobytes()

    wav_audio = AudioSegment(
        audio_pcm16.tobytes(),
        frame_rate=original_sr,
        sample_width=OPENAI_TTS_BIT_DEPTH // 8,
        channels=OPENAI_TTS_CHANNELS,
    )
    if target_sr is not None and target_sr != original_sr:
        wav_audio = wav_audio.set_frame_rate(target_sr)

    # Convert WAV to MP3
    target_io = io.BytesIO()
    wav_audio.export(target_io, format=response_format)
    target_io.seek(0)

    return target_io.getvalue()


def load_voice_presets(state: State,
                       voice_presets_dir: str,
                       interval: int = 10):
    while True:
        try:
            voice_file = os.path.join(voice_presets_dir, "config.json")
            if voice_file is not None:
                with open(voice_file) as f:
                    new_presents = json.load(f)
                diff = set(new_presents.keys()) - set(
                    state.voice_presets.keys())
                if len(diff) > 0:
                    logger.info("New voice presets added: %s", diff)
                state.voice_presets = new_presents
        except Exception as e:
            logger.error("Error loading voice presets: %s", str(e))
            logger.error("Detailed traceback:\n%s", traceback.format_exc())
            time.sleep(interval)


class HiggsAudioServingAudio(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        voice_presets_dir: str,
        chat_template_content_format: ChatTemplateContentFormatOption,
        *,
        audio_tokenizer: Optional[AudioTokenizer] = None,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)
        self.voice_presets_dir = voice_presets_dir
        self.request_logger = request_logger
        self.chat_template_content_format: Final = chat_template_content_format
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)

        self.audio_tokenizer = audio_tokenizer
        self.audio_num_codebooks = self.audio_tokenizer.num_codebooks
        self.audio_codebook_size = self.audio_tokenizer.codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate //
                                     self.audio_tokenizer_tps)
        self.audio_stream_bos_id = model_config.hf_config.audio_stream_bos_id
        self.audio_stream_eos_id = model_config.hf_config.audio_stream_eos_id

    # ruff: noqa: E501  # Disable specific lint rules
    def get_chat_template(self) -> str:
        return (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + "
            "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = bos_token + content %}"
            "{% endif %}"
            "{% if message['role'] == 'assistant' and '<|audio_bos|><|AUDIO|>' in message['content'] %}"
            "{% set content = content.replace('<|audio_bos|><|AUDIO|>', '<|audio_out_bos|><|AUDIO|>') %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>' }}"
            "{% endif %}")

    async def create_audio_speech_stream(
        self,
        request: AudioSpeechRequest,
        voice_presets: dict,
        raw_request: Optional[Request] = None,
    ) -> AsyncGenerator[bytes, None]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = "audiospeech-" \
                     f"{self._base_request_id(raw_request)}"
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            sampling_params = request.to_sampling_params()
            self._log_inputs(request_id,
                             request.input,
                             params=sampling_params,
                             lora_request=None,
                             prompt_adapter_request=None)
            tokenizer = await self.engine_client.get_tokenizer(None)
            engine_prompt = await self.prepare_engine_prompt(
                request, tokenizer, voice_presets)
            generator = self.engine_client.generate(
                engine_prompt,
                sampling_params,
                request_id,
            )
            generators.append(generator)
        except ValueError as e:
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

        return self.audio_speech_stream_generator(request, result_generator)

    async def prepare_engine_prompt(
            self,
            request: AudioSpeechRequest,
            tokenizer: AnyTokenizer,
            voice_presets: Optional[dict] = None) -> str:
        messages = self.prepare_messages(request, voice_presets)
        resolved_content_format = resolve_chat_template_content_format(
            self.get_chat_template(),
            None,
            self.chat_template_content_format,
            tokenizer,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        conversation, mm_data_future = parse_chat_messages_futures(
            messages,
            self.model_config,
            tokenizer,
            content_format=resolved_content_format,
        )

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=self.get_chat_template(),
            add_generation_prompt=True,
            continue_final_message=False,
            tools=None,
            documents=None,
        )

        request_prompt = apply_hf_chat_template(
            tokenizer,
            trust_remote_code=self.model_config.trust_remote_code,
            conversation=conversation,
            **_chat_template_kwargs,
        )

        mm_data = await mm_data_future
        prompt_inputs = await self._tokenize_prompt_input_async(
            request,
            tokenizer,
            request_prompt,
            truncate_prompt_tokens=None,
            add_special_tokens=False,
        )
        engine_prompt = TokensPrompt(
            prompt_token_ids=prompt_inputs["prompt_token_ids"])
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data

        return engine_prompt

    def prepare_messages(
        self,
        request: AudioSpeechRequest,
        voice_presets: Optional[dict] = None
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        audio_base64, reference_text, system_prompt = \
            self.tts_voice_raw(request.voice, self.voice_presets_dir, voice_presets)
        messages.append({
            "role": "system",
            "content": system_prompt or TTS_SYSTEM_PROMPT
        })

        # SFT model uses system prompt instead of reference audio for TTS
        if system_prompt is None:
            messages.append({"role": "user", "content": reference_text})
            messages.append({
                "role":
                "assistant",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    }
                }]
            })

        messages.append({"role": "user", "content": request.input})

        return messages

    def tts_voice_raw(self, voice: str, voice_presets_dir: str,
                      voice_presets: dict):
        if voice not in voice_presets:
            default_voice = list(voice_presets.keys())[0]
            logger.warning("Unsupported voice: %s, using default voice: %s",
                           voice, default_voice)
            voice = default_voice
        path = os.path.join(voice_presets_dir,
                            voice_presets[voice]["audio_file"])
        audio_base64 = encode_base64_content_from_file(path)
        return audio_base64, voice_presets[voice]["transcript"], voice_presets[
            voice].get("system_prompt", None)

    async def audio_speech_stream_generator(
        self,
        request: AudioSpeechRequest,
        result_generator: AsyncIterator[RequestOutput],
    ) -> AsyncGenerator[str, None]:
        prev_resampled_audio = None
        fade_length = int(OPENAI_TTS_SAMPLING_RATE * 0.02)  # 20ms
        # Create fade curves
        fade_out = np.linspace(1, 0, fade_length)
        fade_in = np.linspace(0, 1, fade_length)
        audio_chunk_size = request.audio_chunk_size or self.audio_tokenizer_tps
        audio_chunk_overlap_size = \
            request.audio_chunk_overlap_size or self.audio_tokenizer_tps

        audio_tokens_cache = np.ndarray((0, self.audio_num_codebooks),
                                        dtype=np.int64)
        is_first_audio_chunk = True
        fade_out_audio = None
        finish_reason_sent = False
        previous_num_tokens = 0
        try:
            async for res in result_generator:
                assert len(
                    res.outputs
                ) == 1, "Only one output should be generated per request"
                output = res.outputs[0]

                if finish_reason_sent:
                    continue

                delta_text = output.text
                if not delta_text and not output.token_ids and \
                    not previous_num_tokens:
                    # Chunked prefill case, don't return empty chunks
                    continue

                audio_chunk = None
                if output.mm_token_ids is None:
                    if audio_tokens_cache.shape[0] > 0:
                        audio_chunk, fade_out_audio = create_audio_chunk(
                            audio_tokens_cache,
                            audio_chunk_size,
                            fade_out_audio,
                            finalize=True,
                            audio_tokenizer=self.audio_tokenizer,
                            audio_codebook_size=self.audio_codebook_size,
                            samples_per_token=self.samples_per_token,
                            audio_num_codebooks=self.audio_num_codebooks,
                            audio_stream_bos_id=self.audio_stream_bos_id,
                            audio_stream_eos_id=self.audio_stream_eos_id,
                            return_as_numpy_audio=True)
                        audio_tokens_cache = np.ndarray(
                            (0, self.audio_num_codebooks), dtype=np.int64)
                        fade_out_audio = None
                        # Reset the flag for the next audio sequences
                        is_first_audio_chunk = True
                else:
                    audio_tokens_cache = np.concatenate([
                        audio_tokens_cache,
                        output.mm_token_ids,
                    ],
                                                        axis=0)
                    curr_audio_chunk_size = audio_tokens_cache.shape[0]

                    # The first audio chunk is generated with with less tokens than other chunks
                    # to reduce the first audio latency
                    if is_first_audio_chunk and \
                        curr_audio_chunk_size >= (audio_chunk_size + self.audio_num_codebooks - 1):
                        first_audio_chunk_size = \
                            int(audio_chunk_size - self.audio_num_codebooks + 1)
                        audio_chunk, fade_out_audio = create_audio_chunk(
                            audio_tokens_cache,
                            first_audio_chunk_size,
                            fade_out_audio,
                            finalize=False,
                            audio_tokenizer=self.audio_tokenizer,
                            audio_codebook_size=self.audio_codebook_size,
                            samples_per_token=self.samples_per_token,
                            audio_num_codebooks=self.audio_num_codebooks,
                            audio_stream_bos_id=self.audio_stream_bos_id,
                            audio_stream_eos_id=self.audio_stream_eos_id,
                            return_as_numpy_audio=True)
                        audio_tokens_cache = audio_tokens_cache[
                            first_audio_chunk_size:]
                        is_first_audio_chunk = False
                    elif not is_first_audio_chunk and \
                        curr_audio_chunk_size >= (audio_chunk_size + audio_chunk_overlap_size):
                        audio_chunk, fade_out_audio = create_audio_chunk(
                            audio_tokens_cache,
                            audio_chunk_size,
                            fade_out_audio,
                            finalize=False,
                            audio_tokenizer=self.audio_tokenizer,
                            audio_codebook_size=self.audio_codebook_size,
                            samples_per_token=self.samples_per_token,
                            audio_num_codebooks=self.audio_num_codebooks,
                            audio_stream_bos_id=self.audio_stream_bos_id,
                            audio_stream_eos_id=self.audio_stream_eos_id,
                            return_as_numpy_audio=True)
                        audio_tokens_cache = audio_tokens_cache[
                            audio_chunk_size:]

                    if output.finish_reason is not None:
                        finish_reason_sent = True

                if audio_chunk is not None:
                    output_audio, prev_resampled_audio = self._maybe_upsample_audio(
                        audio_chunk=audio_chunk,
                        prev_resampled_audio=prev_resampled_audio,
                        request=request,
                        fade_length=fade_length,
                        fade_in=fade_in,
                        fade_out=fade_out)
                    yield output_audio

                previous_num_tokens += len(output.token_ids)

        except Exception as e:
            logger.exception("Error in audio speech stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield data

        # Process any remaining audio tokens if any
        if audio_tokens_cache.shape[0] > 0:
            audio_chunk, fade_out_audio = create_audio_chunk(
                audio_tokens_cache,
                audio_chunk_size,
                fade_out_audio,
                audio_tokenizer=self.audio_tokenizer,
                audio_codebook_size=self.audio_codebook_size,
                samples_per_token=self.samples_per_token,
                audio_num_codebooks=self.audio_num_codebooks,
                audio_stream_bos_id=self.audio_stream_bos_id,
                audio_stream_eos_id=self.audio_stream_eos_id,
                finalize=True,
                return_as_numpy_audio=True)
            if audio_chunk is not None:
                output_audio, _ = self._maybe_upsample_audio(
                    audio_chunk=audio_chunk,
                    prev_resampled_audio=prev_resampled_audio,
                    request=request,
                    fade_length=fade_length,
                    fade_in=fade_in,
                    fade_out=fade_out)
                yield output_audio

        # Yield an empty chunk to indicate the end of the stream
        yield b''

    def _maybe_upsample_audio(self, audio_chunk: np.ndarray,
                              prev_resampled_audio: np.ndarray,
                              request: AudioSpeechRequest, fade_length: int,
                              fade_in: np.ndarray, fade_out: np.ndarray):
        needs_upsample = self.audio_tokenizer.sampling_rate != OPENAI_TTS_SAMPLING_RATE
        # Resample if needed
        if needs_upsample:
            current_audio = librosa.resample(
                audio_chunk,
                orig_sr=self.audio_tokenizer.sampling_rate,
                target_sr=OPENAI_TTS_SAMPLING_RATE)
        else:
            current_audio = audio_chunk

        # Apply crossfade if we have a previous chunk and we upsampled
        if prev_resampled_audio is not None and needs_upsample:
            output_audio = \
                self._crossfade_audios(prev_resampled_audio, current_audio,
                                       fade_length, fade_in, fade_out)
            output_audio = pcm_to_target_format_bytes(
                output_audio[:-fade_length],
                response_format=request.response_format,
                original_sr=self.audio_tokenizer.sampling_rate,
                target_sr=OPENAI_TTS_SAMPLING_RATE)
        elif needs_upsample:
            output_audio = pcm_to_target_format_bytes(
                current_audio[:-fade_length],
                response_format=request.response_format,
                original_sr=self.audio_tokenizer.sampling_rate,
                target_sr=OPENAI_TTS_SAMPLING_RATE)
        else:
            # No crossfade needed, just yield the current audio
            output_audio = pcm_to_target_format_bytes(
                current_audio,
                response_format=request.response_format,
                original_sr=self.audio_tokenizer.sampling_rate,
                target_sr=OPENAI_TTS_SAMPLING_RATE)

        return output_audio, current_audio

    def _crossfade_audios(self, prev_audio: np.ndarray, curr_audio: np.ndarray,
                          fade_length: int, fade_in: np.ndarray,
                          fade_out: np.ndarray):
        # Get the overlapping regions
        prev_end = prev_audio[-fade_length:]
        curr_start = curr_audio[:fade_length]

        # Create crossfaded section
        crossfaded = (prev_end * fade_out) + (curr_start * fade_in)

        # Combine previous audio (except fade region) with crossfade and current audio
        output_audio = np.concatenate([crossfaded, curr_audio[fade_length:]])
        return output_audio
