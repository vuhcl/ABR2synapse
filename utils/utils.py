import numpy as np
import pandas as pd
import io
import torch
import re

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.ABRA_35.models import CNN, interpolate_and_smooth


def load_peak_finding_model():
    filter1 = 128
    filter2 = 32
    dropout1 = 0.5
    dropout2 = 0.3
    dropout_fc = 0.1

    # Model initialization
    peak_finding_model = CNN(filter1, filter2, dropout1, dropout2, dropout_fc)
    model_loader = torch.load('./utils/ABRA_35/models/waveI_cnn.pth')
    peak_finding_model.load_state_dict(model_loader)
    peak_finding_model.eval()
    return peak_finding_model


def preprocess_waveform(waveform, time_scale):
    orig_y = waveform

    tenms = int((10/time_scale)*len(orig_y)) if time_scale > 10 else len(orig_y)
    y_values_fpf = interpolate_and_smooth(orig_y[:tenms], 244)

    flattened_data = y_values_fpf.flatten().reshape(-1, 1)

    # Step 1: Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(flattened_data)

    # Step 2: Apply min-max scaling
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = min_max_scaler.fit_transform(standardized_data).reshape(y_values_fpf.shape)

    y_values_fpf = scaled_data

    return interpolate_and_smooth(y_values_fpf)


def peak_finding(wave, peak_finding_model, running_avg_method=True):
    # Prepare waveform
    waveform = interpolate_and_smooth(wave)
    waveform_torch = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Get prediction from model
    pk_outputs = peak_finding_model(waveform_torch)
    prediction = int(round(pk_outputs.detach().numpy()[0][0], 0))

    # Apply Gaussian smoothing
    smoothed_waveform = gaussian_filter1d(wave, sigma=1.0)
    if running_avg_method:
        avg_window = 25
        running_average = np.convolve(smoothed_waveform, np.ones(avg_window)/avg_window, mode='same')
        smoothed_waveform = smoothed_waveform - running_average
    
    # Find peaks and troughs
    n, t, window = 16, 7, 10
    start_point = max(0, prediction - window)
    smoothed_peaks, _ = find_peaks(smoothed_waveform[start_point:], distance=n)
    smoothed_troughs, _ = find_peaks(-smoothed_waveform, distance=t)

    peaks_within_ms = np.array([])
    ms_cutoff = 0.25
    while len(peaks_within_ms) == 0:
        ms_window = int(ms_cutoff * len(smoothed_waveform) / 10)  
        candidate_peaks = smoothed_peaks + start_point
        within_ms_mask = np.abs(candidate_peaks - prediction) <= ms_window
        peaks_within_ms = candidate_peaks[within_ms_mask]
        ms_cutoff += 0.25
    tallest_peak_idx = np.argmax(smoothed_waveform[peaks_within_ms])
    pk1 = peaks_within_ms[tallest_peak_idx]

    peaks = smoothed_peaks + start_point
    peaks = peaks[peaks > pk1]
    sorted_indices = np.argsort(smoothed_waveform[peaks])

    highest_smoothed_peaks = np.sort(np.concatenate(
        ([pk1], peaks[sorted_indices[-min(4, peaks.size):]])
        )) 
    relevant_troughs = np.array([])
    for p in range(len(highest_smoothed_peaks)):
        if p != 4:
            eligible_troughs = smoothed_troughs[(smoothed_troughs > highest_smoothed_peaks[p]) & (smoothed_troughs < highest_smoothed_peaks[p+1])]
        else:
            eligible_troughs = smoothed_troughs[smoothed_troughs > highest_smoothed_peaks[p]]
        if len(eligible_troughs) > 0:
            eligible_trough_depths = smoothed_waveform[eligible_troughs]
            deepest_trough_idx = np.argmin(eligible_trough_depths)
            relevant_troughs = np.append(relevant_troughs, int(eligible_troughs[deepest_trough_idx]))

    relevant_troughs = relevant_troughs.astype('i')
    return highest_smoothed_peaks, relevant_troughs

def extract_metadata(metadata_lines):
    # Dictionary to store extracted metadata
    metadata = {}
    
    for line in metadata_lines:
        # Extract SW FREQ
        freq_match = re.search(r'SW FREQ:\s*(\d+\.?\d*)', line)
        if freq_match:
            metadata['SW_FREQ'] = float(freq_match.group(1))
        
        # Extract LEVELS
        levels_match = re.search(r':LEVELS:\s*([^:]+)', line)
        if levels_match:
            # Split levels and convert to list of floats
            metadata['LEVELS'] = [float(level) for level in levels_match.group(1).split(';') if level]
    
    return metadata

def read_custom_tsv(file_path):
    # Read the entire file
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        content = f.read()
    
    # Split the content into metadata and data sections
    metadata_lines = []
    data_section = None
    
    # Find the ':DATA' marker
    data_start = content.find(':DATA')
    
    if data_start != -1:
        # Extract metadata (lines before ':DATA')
        metadata_lines = content[:data_start].split('\n')
        
        # Extract data section
        data_section = content[data_start:].split(':DATA')[1].strip()
    
    # Extract specific metadata
    metadata = extract_metadata(metadata_lines)
    
    # Read the data section directly
    try:
        # Use StringIO to create a file-like object from the data section
        raw_data = pd.read_csv(
            io.StringIO(data_section), 
            sep='\s+',  # Use whitespace as separator
            header=None
        )
        raw_data = raw_data.T
        # Add metadata columns to the DataFrame
        if 'SW_FREQ' in metadata:
            raw_data['Freq(Hz)'] = metadata['SW_FREQ']
            raw_data['Freq(Hz)'] = raw_data['Freq(Hz)'].apply(lambda x: x*1000)
        
        if 'LEVELS' in metadata:
            # Repeat levels to match the number of rows
            levels_repeated = metadata['LEVELS'] * (len(raw_data) // len(metadata['LEVELS']) + 1)
            raw_data['Level(dB)'] = levels_repeated[:len(raw_data)]
        
        filtered_data = raw_data.apply(pd.to_numeric, errors='coerce').dropna()
        filtered_data.columns = filtered_data.columns.map(str)

        columns = ['Freq(Hz)'] + ['Level(dB)'] + [col for col in filtered_data.columns if col.isnumeric() == True]
        filtered_data = filtered_data[columns]
        return filtered_data
    
    except Exception as e:
        print(f"Error reading data: {e}")
        return None, metadata