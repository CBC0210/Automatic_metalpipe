import librosa
import soundfile as sf
import numpy as np

# 定义音效音量倍率常数
EFFECT_VOLUME = 1.0  # 可以根据需要调整这个值来控制音效音量
# 定义歌曲开头的最小无音效时间点（秒）
MIN_START_TIME = 3.0  # 确保第一个音效至少在3秒之后

# 载入主音乐文件和钢管落地音效样本
main_audio, sr = librosa.load('main_song.wav', sr=None)
pipe_sound, _ = librosa.load('pipe_drop.wav', sr=sr)

print("分析歌曲音乐特征...")

# 综合使用多种特征检测音乐变化点
# 1. 提取节拍信息
tempo, beat_frames = librosa.beat.beat_track(y=main_audio, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(f"检测到节拍，BPM: {float(tempo):.1f}")

# 2. 提取和声特征
harmonic = librosa.effects.harmonic(main_audio)
chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)

# 3. 提取频谱对比度（突变检测）
mfcc = librosa.feature.mfcc(y=main_audio, sr=sr)
mfcc_delta = np.diff(mfcc, axis=1)
mfcc_change = np.sum(np.abs(mfcc_delta), axis=0)

# 4. 计算能量包络变化
energy = librosa.feature.rms(y=main_audio)[0]
energy_delta = np.diff(energy)
energy_change_points = np.where(energy_delta > np.std(energy_delta))[0]
energy_change_times = librosa.frames_to_time(energy_change_points, sr=sr)

# 5. 结合不同特征的变化点
# 计算和声变化（参考原来的方法但更灵敏）
chroma_diff = np.sqrt(np.sum(np.diff(chroma, axis=1)**2, axis=0))
chord_change_points = np.where(chroma_diff > np.mean(chroma_diff) * 0.5)[0]
chord_change_times = librosa.frames_to_time(chord_change_points, sr=sr)

# 6. 使用结构分段作为主要的变化点来源
print("检测音乐结构变化...")
# 使用频谱对比度寻找明显的变化点
try:
    change_points = librosa.segment.agglomerative(S=mfcc, k=20)
    change_times = librosa.frames_to_time(change_points, sr=sr)
except Exception as e:
    print(f"结构分段检测错误，使用备选方法: {e}")
    # 备选方法：使用和声变化作为主要变化点
    change_times = chord_change_times

# 7. 寻找歌曲中的重要节奏变化点和音乐停顿
onset_env = librosa.onset.onset_strength(y=main_audio, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
beat_strength = librosa.util.normalize(onset_env)
strong_beats = np.where(beat_strength > 0.7)[0]  # 只选择强拍点
strong_beat_times = librosa.frames_to_time(strong_beats, sr=sr)

# 8. 结合所有变化点信息，并进行筛选
# 创建用于存储变化点的列表
change_candidates = []

# 使用音乐事件检测(使用onset检测)来定位明显的音频事件
try:
    # 尝试检测音乐中的重要事件(如鼓点、声音的突然开始等)
    onset_frames = librosa.onset.onset_detect(y=main_audio, sr=sr, 
                                             backtrack=True,
                                             units='time')
    # 将这些事件作为基础变化点
    change_candidates.extend(onset_frames)
    print(f"检测到 {len(onset_frames)} 个音乐事件点")
except Exception as e:
    print(f"音乐事件检测错误: {e}")
    # 如果失败，使用节拍点作为备选
    change_candidates.extend(beat_times)

# 添加部分和弦变化点
for ct in chord_change_times:
    if np.random.random() < 0.3:  # 使用30%的和弦变化点
        change_candidates.append(ct)

# 添加节拍中的一部分
for bt in beat_times:
    if np.random.random() < 0.1:  # 只使用10%的节拍点
        change_candidates.append(bt)

# 确保包含结构变化点
change_candidates.extend(change_times)

# 改进人声/歌词检测，更准确地找到歌词段落的第一个音符
try:
    print("尝试检测人声/歌词段落开头...")
    
    # 提取梅尔频谱
    S = librosa.feature.melspectrogram(y=main_audio, sr=sr, n_mels=128, fmax=8000)
    
    # 定义人声频率范围 (通常在中频范围)
    vocal_range = np.s_[20:80]  # 梅尔频谱中可能包含人声的频率范围
    
    # 创建一个掩码来识别人声
    vocal_mask = np.zeros(S.shape[1], dtype=bool)
    
    # 通过计算中频能量与总能量的比率来判断是否有人声
    for t in range(S.shape[1]):
        mid_energy = np.sum(S[vocal_range, t])
        total_energy = np.sum(S[:, t])
        ratio = mid_energy / total_energy if total_energy > 0 else 0
        # 如果中频能量比例较高，可能存在人声
        vocal_mask[t] = ratio > 0.5 and total_energy > np.median(np.sum(S, axis=0))
    
    # 找到所有人声帧
    vocal_frames = np.where(vocal_mask)[0]
    
    # 寻找歌词段落的开头（人声出现的起始点）
    vocal_segment_starts = []
    if len(vocal_frames) > 0:
        # 第一个人声帧一定是一个开始
        vocal_segment_starts.append(vocal_frames[0])
        
        # 寻找人声之间的间隙，间隙后的第一个人声帧是新段落的开始
        for i in range(1, len(vocal_frames)):
            # 如果当前帧与前一帧之间有明显间隙 (大于0.5秒)
            if vocal_frames[i] - vocal_frames[i-1] > int(0.5 * sr / 512):
                vocal_segment_starts.append(vocal_frames[i])
    
    # 将帧索引转换为时间点
    hop_length = 512  # 默认梅尔频谱的跳跃长度
    vocal_start_times = librosa.frames_to_time(vocal_segment_starts, sr=sr, hop_length=hop_length)
    
    print(f"检测到 {len(vocal_start_times)} 个歌词段落开头")
    
    # 将歌词段落开头作为主要添加点
    for vt in vocal_start_times:
        # 为歌词段落开头赋予很高的权重（添加多次以提高采样概率）
        for _ in range(5):  # 添加5次，大幅提高被选中的概率
            change_candidates.append(vt)
except Exception as e:
    print(f"人声段落检测错误: {e}")

# 尝试使用能量变化来辅助检测段落开头
try:
    # 计算音频能量
    energy = librosa.feature.rms(y=main_audio)[0]
    
    # 标准化能量
    energy_norm = librosa.util.normalize(energy)
    
    # 计算能量变化率
    energy_diff = np.diff(energy_norm)
    
    # 找到能量突然增大的点（可能是新段落或新音符的开始）
    energy_peaks = np.where(energy_diff > np.std(energy_diff) * 2)[0]
    
    # 转换为时间
    energy_peak_times = librosa.frames_to_time(energy_peaks, sr=sr, hop_length=512)
    
    print(f"检测到 {len(energy_peak_times)} 个可能的音符/段落开头")
    
    # 将这些点也添加为候选点，但权重低于人声段落开头
    for et in energy_peak_times:
        if np.random.random() < 0.3:  # 30%的概率添加
            change_candidates.append(et)
except Exception as e:
    print(f"能量变化检测错误: {e}")

# 排序并去重
change_candidates = sorted(set(change_candidates))

# 过滤掉开头过早的添加点，只要确保第一个点不是在开头
if change_candidates and change_candidates[0] < MIN_START_TIME:
    # 找到第一个大于等于MIN_START_TIME的点，或者至少移除第一个点
    first_valid_idx = 0
    while first_valid_idx < len(change_candidates) and change_candidates[first_valid_idx] < MIN_START_TIME:
        first_valid_idx += 1
    
    if first_valid_idx < len(change_candidates):
        change_candidates = change_candidates[first_valid_idx:]
    elif len(change_candidates) > 1:  # 如果所有点都太早，至少移除第一个
        change_candidates = change_candidates[1:]

print(f"确保第一个添加点不早于 {MIN_START_TIME} 秒，剩余 {len(change_candidates)} 个候选点")

# 使用动态间隔进行过滤
filtered_times = []
last_time = -2.0  # 初始值设为负数以确保第一个点被包含

for t in change_candidates:
    # 计算当前位置的能量值，用于确定是否是一个重要时刻
    t_sample = int(t * sr)
    if t_sample >= len(main_audio) - 100:
        continue
    
    # 计算当前位置的能量
    window_size = min(2048, len(main_audio) - t_sample)
    if window_size <= 0:
        continue
    
    window = main_audio[t_sample:t_sample+window_size]
    current_energy = np.mean(window**2)
    
    # 基本间隔时间
    base_interval = 0.6  # 默认最小间隔
    
    # 如果是低能量区域(可能是歌曲的间奏或过渡)，使用较长间隔
    if current_energy < 0.01:
        dynamic_interval = base_interval * 1.5
    else:
        dynamic_interval = base_interval
    
    # 使用伪随机化间隔，使添加点听起来更自然
    random_factor = 0.7 + np.random.random() * 0.6  # 0.7-1.3的随机因子
    dynamic_interval *= random_factor
    
    # 确保间隔在合理范围内
    dynamic_interval = max(0.3, min(1.2, dynamic_interval))
    
    if t - last_time >= dynamic_interval:
        filtered_times.append(t)
        last_time = t

change_candidates = filtered_times

# 限制总数量，但确保歌词段落开头被优先保留
max_changes = 60  # 减少总数量以避免过多音效
if len(change_candidates) > max_changes:
    # 尝试识别哪些点是歌词段落开头
    vocal_points = set(vocal_start_times) if 'vocal_start_times' in locals() else set()
    
    # 确保所有歌词段落开头都被保留
    kept_points = []
    for t in change_candidates:
        # 检查是否是歌词段落开头，或者与某个歌词段落开头非常接近
        is_vocal_point = False
        for vt in vocal_points:
            if abs(t - vt) < 0.1:  # 如果在0.1秒内很接近
                is_vocal_point = True
                break
        
        if is_vocal_point:
            kept_points.append(t)
    
    # 如果已保留的点不够，从其他点中均匀选择剩余的
    remaining_slots = max_changes - len(kept_points)
    if remaining_slots > 0:
        remaining_points = [p for p in change_candidates if p not in kept_points]
        if remaining_points:
            indices = np.linspace(0, len(remaining_points) - 1, min(remaining_slots, len(remaining_points)), dtype=int)
            kept_points.extend([remaining_points[i] for i in indices])
    
    # 确保点是有序的
    change_candidates = sorted(kept_points)

print(f"最终确定 {len(change_candidates)} 个添加点，其中优先保留歌词段落开头")

# 处理每个变化点
output_audio = main_audio.copy()

for t in change_candidates:
    # 计算添加位置对应的采样点索引
    start_sample = int(t * sr)
    
    # 如果起始点太靠近歌曲末尾，跳过此次添加
    if start_sample + 1000 >= len(main_audio):
        continue
    
    # 获取当前位置的音高信息
    # 使用小窗口从当前位置的音频片段中提取音高
    segment = main_audio[start_sample:start_sample + int(0.2 * sr)]  # 取200ms的片段
    if len(segment) == 0:
        continue
        
    # 提取主导音高并添加随机变化以增加音高多样性
    try:
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
        if pitches.size == 0 or magnitudes.size == 0:
            shift = np.random.uniform(-1, 1)  # 随机音高
        else:
            # 找到每一帧中最强的音高
            pitch_indices = magnitudes.argmax(axis=0)
            pitches_per_frame = [pitches[i, t] for t, i in enumerate(pitch_indices) if i < pitches.shape[0]]
            if not pitches_per_frame:
                shift = np.random.uniform(-1, 1)
            else:
                # 计算平均音高
                avg_pitch = np.mean([p for p in pitches_per_frame if p > 0])
                
                # 计算基础音高调整值
                if avg_pitch > 0:
                    midi_note = librosa.hz_to_midi(avg_pitch)
                    # 映射到-2到+2的小范围
                    shift_base = max(-1.5, min(1.5, (round(midi_note) - 60) / 12))
                    # 添加小的随机变化，使声音更有变化
                    shift = shift_base + np.random.uniform(-0.5, 0.5)
                    # 确保在范围内
                    shift = max(-2, min(2, shift))
                else:
                    shift = np.random.uniform(-1, 1)
    except Exception as e:
        print(f"音高检测错误: {e}")
        # 发生错误时使用随机音高
        shift = np.random.uniform(-1, 1)
    
    print(f"位置 {t:.2f}秒, 音高调整: {shift:.2f} 半音")
    
    # 将钢管音效进行 pitch-shift
    shifted_pipe = librosa.effects.pitch_shift(y=pipe_sound, sr=sr, n_steps=shift)
    
    end_sample = start_sample + len(shifted_pipe)
    
    # 混合音频
    if end_sample < len(output_audio):
        # 创建淡入淡出窗口
        fade_length = min(2000, len(shifted_pipe))
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        # 应用淡入淡出
        fade_pipe = shifted_pipe.copy()
        fade_pipe[:fade_length] *= fade_in
        fade_pipe[-fade_length:] *= fade_out
        
        # 根据当前音量动态调整混合音量
        local_energy = np.mean(main_audio[start_sample:end_sample]**2)
        # 在安静的部分稍微增加效果音量，在响亮的部分保持正常
        volume_factor = 1.2 if local_energy < 0.01 else 1.0
        
        # 应用音效音量倍率
        final_volume = volume_factor * EFFECT_VOLUME
        
        # 添加到输出音频
        output_audio[start_sample:end_sample] += fade_pipe * final_volume
    else:
        # 如果超出长度，则截断
        fade_pipe = shifted_pipe[:len(output_audio)-start_sample].copy()
        if len(fade_pipe) > 0:
            fade_length = min(2000, len(fade_pipe))
            fade_in = np.linspace(0, 1, fade_length)
            if len(fade_pipe) > fade_length:
                fade_pipe[:fade_length] *= fade_in
            # 应用音效音量倍率
            output_audio[start_sample:] += fade_pipe * EFFECT_VOLUME

print("混合完成，保存输出文件...")
# 导出混音后的音频
sf.write('mixed_song.wav', output_audio, sr)
print(f"处理完成，输出文件已保存为 mixed_song.wav (音效音量倍率: {EFFECT_VOLUME}，优先添加在歌词段落开头)")
