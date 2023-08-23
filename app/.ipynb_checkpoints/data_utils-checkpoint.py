import pyedflib
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime


def load_data(db_folder):
    """
    Read the data from db_folder and preprpocess it.
    
    Parameters:
    -----------
    - db_folder: str
        Path towards the data
        
    Returns
    -------
    - raw_records: nmupy.array of shape (n_records, signal_len, n_signals)
        Array containing recorded data for all invidual presented in the study
    
    - raw_respevents: nmupy.array of shape (n_records, signal_len, 1)
        Labels of record signals
    
    """
    print(f"Loading data....")
    
    #file containing details about subjects
    subject_details = pd.read_excel(os.path.join(db_folder, "SubjectDetails.xls"))
    
    #number of records or subjects
    n_records = len(subject_details)
    
    #Study duration. Set to constant for all subjects to ease data prpocessing
    duration = subject_details['Study Duration (hr)'].min()*3600
    
    #Reading records
    signals, signal_headers, header = pyedflib.highlevel.read_edf(os.path.join(db_folder,"ucddb002.rec"))
    signal_headers_table = pd.DataFrame(signal_headers)
    n_signals = len(signal_headers_table)
    sample_rate = signal_headers_table['sample_rate'].max()
    signal_len = int(sample_rate*duration)
    
    raw_records = np.zeros((n_records, signal_len, n_signals), dtype="float32")
    raw_respevents = np.zeros((n_records, signal_len, 1), dtype="bool")
    
    for entry in tqdm(os.scandir(db_folder)):
        rootname, ext = os.path.splitext(entry.name)
        study_number = rootname[:8].upper()
        if not study_number.startswith("UCDDB"):
            continue
        subject_index, = np.where((subject_details["Study Number"] == study_number))[0]
        if ext == ".rec":
            signals, signal_headers, header = pyedflib.highlevel.read_edf(entry.path)
            for sig, sig_hdr in zip(signals, signal_headers):
                try:
                    signal_index, = np.where((signal_headers_table["label"] == sig_hdr["label"]))[0]
                except ValueError:
                    if sig_hdr["label"] == "Soud":
                          signal_index = 7
                if sig_hdr["sample_rate"] != 128:
                    q = int(sample_rate//sig_hdr["sample_rate"])
                    sig = np.repeat(sig, q)
                    sig = sig[:signal_len]
                    raw_records[subject_index,:,signal_index] = sig.astype("float32")
        elif rootname.endswith("respevt"):
            respevents = pd.read_fwf(os.path.join(db_folder,"ucddb002_respevt.txt"), widths=[10,10,8,9,8,8,6,7,7,5], skiprows=[0,1,2], 
                                     skipfooter=1, engine='python', names=["Time", "Type", "PB/CS", "Duration", "Low", "%Drop", "Snore", "Arousal", "Rate", "Change"])
            respevents["Time"] = (pd.to_datetime(respevents["Time"],format='mixed') - pd.to_datetime(subject_details.loc[subject_index, "PSG Start Time"],
                                                                                                     format='mixed')).astype("timedelta64[s]")%(3600*24)
            respevents["Time"] = pd.to_timedelta(respevents["Time"], unit="s")
            for _, event in respevents.iterrows():
                onset = int(sample_rate*event["Time"].total_seconds())
                offset = onset + int(sample_rate*event["Duration"])
                raw_respevents[subject_index, onset:offset] = 1
                
    return raw_records, raw_respevents


def build_data(db_folder,period):
    """
    Prepapre the data into episode of period seconds
    
    Parameters:
    -----------
    -db_folder:str
        path towards the dataset
        
    - period: int
     Episode duration in seconds (32,64,128,etc)
     
    Returns:
    --------
    - X: np.array of shape (n_records*(signal_len//period),period,n_signals)
    
    - y: np.array of shape (n_records*(signal_len//period),period,1)
    
    """
    raw_records, raw_respevents = load_data(db_folder)
    print(f"Preparing data into periods of {period} seconds....")
    x = []
    y = []
    n_records = raw_records.shape[0]
    for n in range(n_records):
        x_data = raw_records[n]
        y_data = raw_respevents[n]
        for i in range(0,len(x_data),period):
            x+=[x_data[i:i+period]]
            y+=[y_data[i:i+period]]
        if len(x_data)%period!=0:
            x = x[:-1]
            y = y[:-1]
    return np.array(x),np.array(y)
