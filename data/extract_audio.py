# -*- coding: utf-8 -*-
"""
音频特征管线:
1. 从 Slide.zip 的每个 mcz 提取音频 (ogg/mp3/...) 到 data/audio/{song_id}.ogg
2. 用 librosa 计算全曲 mel 谱并缓存 data/audio/{song_id}_mel.npy
3. 生成 data/audio/meta.json 记录映射

之后谱面侧按拍对齐采样 (bpms 表 -> 拍->秒 -> mel 帧插值)。
"""
import io
import json
import os
import sys
import zipfile

import librosa
import numpy as np

ZIP_PATH = "Sort By mode/Slide.zip"
OUT_DIR = "data/audio"
SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {}
    n_ok = 0
    n_skip = 0
    with zipfile.ZipFile(ZIP_PATH) as outer:
        for name in outer.namelist():
            if not name.endswith(".mcz"):
                continue
            # 读 mcz 内第一个 .mc 拿 song_id
            inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
            song_id = None
            audio_name = None
            for mc_name in inner.namelist():
                if mc_name.endswith(".mc"):
                    try:
                        c = json.loads(inner.read(mc_name))
                        sid = c.get("meta", {}).get("song", {}).get("id")
                        if sid is not None:
                            song_id = sid
                    except Exception:
                        pass
                if audio_name is None and mc_name.lower().endswith((".ogg", ".mp3", ".m4a", ".wav", ".flac")):
                    audio_name = mc_name
            if song_id is None or audio_name is None:
                n_skip += 1
                continue
            out_audio = os.path.join(OUT_DIR, f"{song_id}{os.path.splitext(audio_name)[1].lower()}")
            out_mel = os.path.join(OUT_DIR, f"{song_id}_mel.npy")
            if os.path.exists(out_mel):
                meta[str(song_id)] = {"mcz": name, "audio": out_audio, "mel": out_mel, "sr": SR, "hop": HOP, "n_fft": N_FFT, "n_mels": N_MELS}
                n_ok += 1
                continue
            # 提取音频
            raw = inner.read(audio_name)
            with open(out_audio, "wb") as f:
                f.write(raw)
            # mel 谱
            try:
                y, sr = librosa.load(out_audio, sr=SR, mono=True)
            except Exception as e:
                print(f"!! {name}: 音频读取失败 {e}")
                n_skip += 1
                continue
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                power=2.0)
            mel_db = librosa.power_to_db(mel, ref=np.max).T  # (T, 128)
            np.save(out_mel, mel_db.astype(np.float32))
            meta[str(song_id)] = {"mcz": name, "audio": out_audio, "mel": out_mel,
                                  "sr": SR, "hop": HOP, "n_fft": N_FFT, "n_mels": N_MELS,
                                  "audio_frames": int(mel_db.shape[0]), "duration_s": float(len(y) / sr)}
            n_ok += 1
            print(f"[{n_ok}] song {song_id} <- {audio_name} frames={mel_db.shape[0]}")
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"\n完成: {n_ok} 首, 跳过 {n_skip}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
