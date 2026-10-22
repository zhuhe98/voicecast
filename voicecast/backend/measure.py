"""单句声学测量（Praat / librosa）：用于硬检查（时长、静音、爆音）与音高范围检查。

属性：F0（Praat 自相关，75–1000 Hz，适应高音女声）、F0 起伏、频谱重心、语速、HNR、CPPS、F1–F3、响度。
共振峰与响度只在有声帧上取值；每个属性先逐句计算，再取各句中位数。
"""
import re

import librosa
import numpy as np
import parselmouth
import zhconv
from parselmouth.praat import call

PITCH_FLOOR, PITCH_CEILING = 75, 1000
MAX_FORMANT = 5500  # Praat 对女声的常用上限
ATTRS = ["f0_med_st", "f0_std_st", "centroid", "sps", "hnr", "cpps", "f1", "f2", "f3", "loudness"]


def norm_text(s):
    return re.sub(r"[^\w]", "", zhconv.convert(s, "zh-cn")).lower()


def st(hz):
    """Hz → 以 100 Hz 为基准的半音。"""
    return 12 * np.log2(np.asarray(hz) / 100.0)


def clip_attrs(path, text=None):
    """单句：属性 + 硬检查用的原始量。"""
    y, sr = librosa.load(path, sr=16000)
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    times = pitch.xs()[voiced]
    f0 = f0[voiced]

    fm = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=MAX_FORMANT)
    formants = [np.nanmedian([fm.get_value_at_time(k, t) for t in times]) if len(times) else np.nan
                for k in (1, 2, 3)]
    inten = snd.to_intensity(minimum_pitch=PITCH_FLOOR)
    loud = [inten.get_value(t) for t in times]
    cep = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)

    frame_db = librosa.amplitude_to_db(librosa.feature.rms(y=y)[0], ref=1.0)
    dur = len(y) / sr
    return {
        "f0_med_hz": float(np.median(f0)) if len(f0) else np.nan,
        "f0_med_st": float(st(np.median(f0))) if len(f0) else np.nan,
        "f0_std_st": float(np.std(st(f0))) if len(f0) > 1 else np.nan,
        "centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        "sps": len(norm_text(text)) / dur if text else np.nan,
        "hnr": float(call(snd.to_harmonicity_cc(), "Get mean", 0, 0)),
        "cpps": float(call(cep, "Get CPPS", "no", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0,
                           "Straight", "Robust")),  # "Exponential decay"/"Robust slow" 每句约 44 s
        "f1": float(formants[0]), "f2": float(formants[1]), "f3": float(formants[2]),
        "loudness": float(np.nanmean(loud)) if loud else np.nan,
        # 硬检查用
        "dur_s": round(dur, 2), "voiced_frac": round(float(voiced.mean()), 3),
        "finite": bool(np.isfinite(y).all()), "clip_frac": float((np.abs(y) >= 0.999).mean()),
        "silence_frac": round(float((frame_db < -50).mean()), 3),
    }
