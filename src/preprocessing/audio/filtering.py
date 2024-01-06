from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import scipy
import os
import librosa
import librosa.display

load_dotenv()
VIDEO_DATA_PATH = os.getenv("VIDEO_DATA_PATH")
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")


def convert_video_to_audio_moviepy(video_file, output_ext="mp3"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")


def detect_audio_events(mp3_file_path, start_time, end_time):
    y, sr = librosa.load(mp3_file_path, offset=start_time, duration=end_time - start_time)
    y_abs = np.abs(y)
    window_time = int(0.19 * sr)
    mean_y = scipy.signal.medfilt(y_abs, kernel_size=window_time)
    return mean_y, sr


def detect_audio_segments(audio_signal, sr, threshold=0.001):
    is_speaking = audio_signal > threshold
    changes = np.where(is_speaking[:-1] != is_speaking[1:])[0] + 1
    changes_seconds = changes / sr
    start_times = changes[::2]
    end_times = changes[1::2]

    plt.plot(np.arange(len(audio_signal)), audio_signal)
    plt.plot(start_times, np.zeros_like(start_times), marker="o", color="red")
    plt.plot(end_times, np.zeros_like(end_times), marker="o", color="red")
    plt.show()

    audio_segments = {"audio": list(zip(start_times, end_times))}
    print(audio_segments)
    return audio_segments


video_path = f"{VIDEO_DATA_PATH}/p1.mp3"
start_time = 28 * 60 + 37  # 28:37
end_time = start_time + 122

filtered_signal, sr = detect_audio_events(video_path, start_time, end_time)
detect_audio_segments(filtered_signal, sr)
