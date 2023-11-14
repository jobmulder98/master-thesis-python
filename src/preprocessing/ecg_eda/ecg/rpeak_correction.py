import numpy as np
from numpy import typing as npt


def correct_rpeaks(peak_indices):
    peak_indices_diff = np.diff(peak_indices)
    threshold = 200
    corrected_peaks_indices = [peak_indices_diff[0]]
    skip_one_iteration = False
    for i in range(1, len(peak_indices_diff) - 1):
        if not skip_one_iteration:
            previous_peak = np.mean(corrected_peaks_indices)
            current_peak = peak_indices_diff[i]
            next_peak = peak_indices_diff[i+1]
            difference = np.abs(current_peak - previous_peak)
            if difference >= threshold:
                if (current_peak + next_peak - previous_peak) <= threshold:
                    corrected_peaks_indices.append(current_peak + next_peak)
                    skip_one_iteration = True
                elif np.abs(current_peak - 2*next_peak) <= threshold:
                    corrected_peaks_indices.append(current_peak // 2)
                    if current_peak % 2 == 0:
                        corrected_peaks_indices.append(current_peak // 2)
                    else:
                        corrected_peaks_indices.append((current_peak // 2) + 1)
                elif current_peak + next_peak - 2*previous_peak <= threshold:
                    corrected_peaks_indices.append((current_peak + next_peak) // 2)
                    corrected_peaks_indices.append((current_peak + next_peak) // 2)  # Do this two times
                    skip_one_iteration = True
                else:
                    corrected_peaks_indices.append(current_peak)
            else:
                corrected_peaks_indices.append(current_peak)
        else:
            skip_one_iteration = False
    corrected_peaks_indices.append(peak_indices_diff[-1])
    corrected_peaks_indices.insert(0, peak_indices[0])
    return np.cumsum(corrected_peaks_indices)


def correct_rpeaks_manually(participant, condition, corrected_rpeaks: npt.NDArray) -> npt.NDArray:
    if participant == 11 and condition == 2:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 10176))
        corrected_rpeaks = np.append(corrected_rpeaks, 9996)
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 10789))
        corrected_rpeaks = np.append(corrected_rpeaks, 10698)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 11 and condition == 5:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 5491))
        corrected_rpeaks = np.append(corrected_rpeaks, 5303)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 15 and condition == 2:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 11942))
        corrected_rpeaks = np.append(corrected_rpeaks, 11788)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 15 and condition == 4:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 21924))
        corrected_rpeaks = np.append(corrected_rpeaks, 21722)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 16 and condition == 6:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 16540))
        corrected_rpeaks = np.append(corrected_rpeaks, 16315)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 21 and condition == 1:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 84158))
    if participant == 21 and condition == 2:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 108456))
    if participant == 21 and condition == 5:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 5074))
        corrected_rpeaks = np.append(corrected_rpeaks, 5003)
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 101358))
        corrected_rpeaks = np.append(corrected_rpeaks, 101217)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 21 and condition == 6:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 28986))
        corrected_rpeaks = np.append(corrected_rpeaks, 28850)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 21 and condition == 7:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 81422))
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 109054))
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 109559))
        corrected_rpeaks = np.append(corrected_rpeaks, 81340)
        corrected_rpeaks = np.append(corrected_rpeaks, 109321)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 22 and condition == 2:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 56831))
    if participant == 22 and condition == 4:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 114480))
        corrected_rpeaks = np.append(corrected_rpeaks, 114440)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    if participant == 22 and condition == 7:
        corrected_rpeaks = np.delete(corrected_rpeaks, np.where(corrected_rpeaks == 44977))
        corrected_rpeaks = np.append(corrected_rpeaks, 44902)
        corrected_rpeaks = np.sort(corrected_rpeaks)
    return corrected_rpeaks


def check_for_corrupted_data(participant: int, condition: int, signal: npt.NDArray) -> npt.NDArray:
    if participant == 22 and condition == 6:
        signal = signal[:55000]
    return signal
