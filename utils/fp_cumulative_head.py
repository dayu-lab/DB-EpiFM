# fp_cumulative_head.py
import numpy as np


def firing_power(binary_seq: np.ndarray, tau: int) -> np.ndarray:
    """
    计算 firing power:
        fp[n] = 最近 tau 个窗口中被预测为 preictal 的比例

    参数:
        binary_seq: 长度为 T 的 0/1 数组, 例如 O[k] = 1 表示第 k 个窗预测为 preictal
        tau: 以“窗口个数”为单位的滑动窗口长度 (对应 SOP, 2*SOP, 3*SOP)

    返回:
        fp: 长度为 T 的 firing power 序列
    """
    assert binary_seq.ndim == 1
    T = binary_seq.shape[0]
    fp = np.zeros(T, dtype=float)

    cumsum = np.cumsum(binary_seq.astype(float))
    for n in range(T):
        if n < tau:
            fp[n] = cumsum[n] / (n + 1)
        else:
            fp[n] = (cumsum[n] - cumsum[n - tau]) / tau
    return fp


def alarm_generation(fp: np.ndarray, threshold: float, refractory_win: int) -> np.ndarray:
    """
    根据 firing power 生成事件级 alarm 序列 (类似 Batista 源码 alarm_generation).

    参数:
        fp: firing power 序列, 长度 T
        threshold: 触发 alarm 的阈值
        refractory_win: 不应期, 单位为“窗口个数”

    返回:
        alarm: 0/1 数组, alarm[n] = 1 表示在第 n 个窗触发一次 alarm
    """
    T = fp.shape[0]
    alarm = np.zeros(T, dtype=int)
    last_alarm_idx = -10 ** 9

    for n in range(T):
        if fp[n] >= threshold and (n - last_alarm_idx) >= refractory_win:
            alarm[n] = 1
            last_alarm_idx = n

    return alarm


def compute_cumulative_firing_power(
    binary_seq: np.ndarray,
    win_sec: float,
    SOP_min: float,
):
    """
    按论文定义计算 Cumulative FP 所需的三条 FP 曲线 + 累积 FP:

        Event1(最早): 窗口长度 = 3 * SOP
        Event2(中间): 窗口长度 = 2 * SOP
        Event3(最近): 窗口长度 = 1 * SOP
        final_firing_power = fp1 + fp2 + fp3

    参数:
        binary_seq: 0/1 数组 O[k], 长度 T
        win_sec   : 每个窗口的实际时长 (秒), 比如你现在是 10s window, 就是 10
        SOP_min   : 单个 SOP 长度 (分钟), 比如 30 分钟

    返回:
        fp1, fp2, fp3, fp_cum, tau1, tau2, tau3
    """
    O = binary_seq.astype(int)

    # 每个 SOP 对应的窗口个数
    tau3 = int(SOP_min * 60.0 / win_sec)      # 最近事件 (1*SOP)
    tau2 = 2 * tau3                           # 中期事件 (2*SOP)
    tau1 = 3 * tau3                           # 最早事件 (3*SOP)

    fp3 = firing_power(O, tau3)
    fp2 = firing_power(O, tau2)
    fp1 = firing_power(O, tau1)

    fp_cum = fp1 + fp2 + fp3

    return fp1, fp2, fp3, fp_cum, tau1, tau2, tau3


def construct_cumulative_alarms(
    fp_cum: np.ndarray,
    SOP_min: float,
    win_sec: float,
    thresholds=(0.5, 1.0, 1.5),
    refractory_factor=1.0,
):
    """
    仿照 Batista 的 construct_alarm_C, 从 cumulative FP 生成最终 alarm 序列。

    思路:
        1) 对 cumulative FP 分别用三个阈值生成三条 alarm 序列:
               alarm_events[0] -> threshold 0.5 (事件1)
               alarm_events[1] -> threshold 1.0 (事件2)
               alarm_events[2] -> threshold 1.5 (事件3)
        2) 按时间扫描:
               在 SOP 时间窗内依次找到 event1 -> event2 -> event3,
               若三者都出现, 则在 event3 对应的时间点标记一次最终 alarm。

    参数:
        fp_cum: 累积 firing power 序列, 长度 T
        SOP_min: SOP (分钟)
        win_sec: 窗长 (秒)
        thresholds: 三个事件的阈值, 默认 (0.5, 1.0, 1.5)
        refractory_factor: 生成 alarm 时的不应期倍数,
                           默认 1.0 表示不应期 = 1 * SOP (以窗口个数计)

    返回:
        final_alarm: 0/1 数组, 表示最终触发的 alarm 位置
        alarm_events: [alarm_ev1, alarm_ev2, alarm_ev3]
    """
    T = fp_cum.shape[0]
    SOP_win = int(SOP_min * 60.0 / win_sec)        # 一个 SOP 包含的窗口个数
    refractory_win = int(refractory_factor * SOP_win)

    # --- 步骤 1: 对三个阈值分别生成事件级 alarm ---
    alarm_events = []
    for th in thresholds:
        alarm = alarm_generation(fp_cum, threshold=th, refractory_win=refractory_win)
        alarm_events.append(alarm)

    alarm_ev1, alarm_ev2, alarm_ev3 = alarm_events

    # --- 步骤 2: 串联三个事件生成最终 alarm ---
    final_alarm = np.zeros(T, dtype=int)
    i = 0
    while i < T:
        if alarm_ev1[i] == 1:
            # 在 [i, i + SOP_win] 搜索 event2
            ev2_idx = -1
            j_end = min(T, i + SOP_win + 1)
            j = i
            while j < j_end:
                if alarm_ev2[j] == 1:
                    ev2_idx = j
                    break
                j += 1

            if ev2_idx != -1:
                # 在 [ev2_idx, ev2_idx + SOP_win] 搜索 event3
                ev3_idx = -1
                k_end = min(T, ev2_idx + SOP_win + 1)
                k = ev2_idx
                while k < k_end:
                    if alarm_ev3[k] == 1:
                        ev3_idx = k
                        break
                    k += 1

                if ev3_idx != -1:
                    # 三个事件按时间顺序都出现了 -> 在 event3 位置记一次最终 alarm
                    final_alarm[ev3_idx] = 1
                    # 跳到该 alarm 之后继续搜索
                    i = ev3_idx + 1
                    continue

        i += 1

    return final_alarm, alarm_events


def cumulative_fp_head(
    window_probs: np.ndarray,
    win_sec: float,
    SOP_min: float,
    prob_threshold: float = 0.5,
    fp_thresholds=(0.5, 1.0, 1.5),
    refractory_factor=1.0,
):
    """
    从 “窗级 preictal 概率序列” 一步得到 Cumulative FP 的最终 alarm 序列。

    参数:
        window_probs: 长度 T 的概率数组, p[k] = 模型预测第 k 个窗为 preictal 的概率
        win_sec     : 每个窗口对应的秒数 (你现在是 10s 就传 10)
        SOP_min     : 单个 SOP 的长度 (分钟), 如 30
        prob_threshold: 把概率二值化的阈值 (默认 0.5)
        fp_thresholds : cumulative FP 的三个阈值, (0.5,1.0,1.5)
        refractory_factor: 生成事件级 alarm 时的不应期倍数

    返回:
        final_alarm: 0/1 数组, cumulative FP 的最终报警
        fp_tuple   : (fp1, fp2, fp3, fp_cum)
        alarm_events: [alarm_ev1, alarm_ev2, alarm_ev3]
    """
    # 先把概率变成 0/1 序列
    O = (window_probs >= prob_threshold).astype(int)

    # 计算三条 FP + 累积 FP
    fp1, fp2, fp3, fp_cum, tau1, tau2, tau3 = compute_cumulative_firing_power(
        O, win_sec=win_sec, SOP_min=SOP_min
    )

    # 用 cumulative FP 生成最终 alarm
    final_alarm, alarm_events = construct_cumulative_alarms(
        fp_cum=fp_cum,
        SOP_min=SOP_min,
        win_sec=win_sec,
        thresholds=fp_thresholds,
        refractory_factor=refractory_factor,
    )

    return final_alarm, (fp1, fp2, fp3, fp_cum), alarm_events
