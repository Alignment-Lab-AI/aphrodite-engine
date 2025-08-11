# SPDX-License-Identifier: Apache-2.0
import time
from io import BytesIO

import click
import numpy as np
import openai
import soundfile as sf
from pydub import AudioSegment

OPENAI_TTS_SAMPLING_RATE = 24000


@click.command()
@click.option("--api-base",
              type=str,
              default="http://localhost:8000/v1",
              help="API base URL")
@click.option("--api-key", type=str, default="EMPTY", help="API key")
@click.option("--voice-preset",
              type=str,
              default="en_woman_1",
              help="en_woman_1, en_man_1, zh_man_1, hogwarts_v2")
@click.option("--response-format", type=str, default="pcm", help="pcm, wav")
def main(api_base: str, api_key: str, voice_preset: str, response_format: str):
    client = openai.OpenAI(base_url=api_base, api_key=api_key)
    input_text = ("Mr. Bounce was very small and like a rubber ball. "
                  "He loved bouncing around the neighborhood, "
                  "bringing joy to all the children who saw him.")
    models = client.models.list()
    model = models.data[0].id
    start_time = time.time()

    # Collect all audio bytes
    audio_data = b''
    first_audio_latency = None
    with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice_preset,
            response_format=response_format,
            input=input_text,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            audio_data += chunk
            if first_audio_latency is None:
                first_audio_latency = time.time() - start_time

    print(f"First audio latency: {first_audio_latency * 1000} ms")
    print(f"Total time: {(time.time() - start_time) * 1000} ms")

    if response_format == "pcm":
        # Decode base64 data
        # Convert to numpy array with correct data type (16-bit PCM)
        numpy_array = np.frombuffer(audio_data, dtype=np.int16)
        # Write to WAV file with the correct sample rate
        sf.write("output_audio_speech.wav", numpy_array,
                 OPENAI_TTS_SAMPLING_RATE)
    elif response_format == "mp3":
        audio = AudioSegment.from_file(BytesIO(audio_data),
                                       format=response_format)
        audio.export("output_audio_speech.mp3", format=response_format)


if __name__ == "__main__":
    main()
