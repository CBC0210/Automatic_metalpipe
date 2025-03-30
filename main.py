import librosa
import soundfile as sf
import numpy as np

# 载入主音乐文件和钢管落地音效样本
main_audio, sr = librosa.load('main_song.wav', sr=None)
pipe_sound, _ = librosa.load('pipe_drop.wav', sr=sr)

# 定义在歌曲中的添加时间点（单位：秒），例如：节奏变化处
insert_points = [30, 60, 90]  # 可根据具体分析进行调整

# 定义对应的音高调整（单位：半音），可对应不同情绪
pitch_shifts = [-3, 0, 2]  # 例如：第一个音效下调3半音，第二个不变，第三个上调2半音

# 处理每个添加点
output_audio = main_audio.copy()
for t, shift in zip(insert_points, pitch_shifts):
    # 将钢管音效进行 pitch-shift
    shifted_pipe = librosa.effects.pitch_shift(y=pipe_sound, sr=sr, n_steps=shift)
    # 计算添加位置对应的采样点索引
    start_sample = int(t * sr)
    end_sample = start_sample + len(shifted_pipe)
    # 混合音频（简单加和混合，边界处可做淡入淡出处理）
    if end_sample < len(output_audio):
        output_audio[start_sample:end_sample] += shifted_pipe
    else:
        # 如果超出长度，则截断
        output_audio[start_sample:] += shifted_pipe[:len(output_audio)-start_sample]

# 导出混音后的音频
sf.write('mixed_song.wav', output_audio, sr)
