from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import matplotlib.pyplot as plt
# import os
import uvicorn

# data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# signal processing
from scipy import signal
from scipy.ndimage import label
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.integrate import trapz

app = FastAPI()
file_path = "./data/heart.csv"
app.mount('/data', StaticFiles(directory="data"), name="data")
# Mount the "images" folder to be served as static files
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get('/')
async def home():
    return {"Heart Uno"}

@app.post("/upload")
async def add_analysis(csv: UploadFile=File(...)):

    # text = secure_filename(csv.filename)
    contents = csv.file.read()
    with open(f"./data/heart.csv", 'wb') as f:
        f.write(contents)
        
    response = {
        'status': "success"
    }
    
    return response

@app.get("/original")
async def original_signal():
    file_path = "./data/heart.csv"
    ecg = setup(file_path)
    
    original = ecg.hart[0:1000]
    
    ori = [o for o in original]
    
    response = {
        'ecg': ori
    }
    
    encoder = jsonable_encoder(response)
    
    return JSONResponse(content= encoder)

@app.get("/deteksi")
async def deteksi_r():
    ecg = setup(file_path)
    
    sampfrom = 0
    sampto = 3000
    cond_slice = (ecg.index >= sampfrom) & (ecg.index < sampto)
    ecg_slice = ecg.hart[cond_slice]
    
    # deteksi puncak
    peaks, similarity = peaks_similarity(ecg_slice)
    
    image = './images/deteksi.jpg'
    
        # plot similarity
    plt.figure(figsize=(20, 10))

    plt.subplot(211,  facecolor='white')
    plt.title("Sinyal EKG dengan Puncak R")
    plt.plot(ecg_slice, label="EKG", color="#51A6D8")
    plt.plot(peaks, np.repeat(600, peaks.shape[0]), markersize=10, label="R", color="orange", marker="o", linestyle="None")
    plt.legend(loc="upper right")
    plt.tick_params(axis='both')
    plt.xlabel("Waktu (ms)")
    plt.ylabel("Amplitude")

    plt.subplot(212,  facecolor='white')
    # plt.title('Similarity with QRS template', fontsize=24)
    plt.plot(similarity, label="Kesamaan dengan Filter QRS", color="olive")
    plt.legend(loc="upper right")
    plt.tick_params(axis='both')
    plt.xlabel("Waktu (ms)")
    plt.ylabel("Kesamaan (dinormalisasi)")
    
    plt.savefig(image)
    plt.close()
    
    response = {
    'image': 'http://localhost:5000/images/deteksi.jpg',
    }
    
    encoder = jsonable_encoder(response)
    
    return JSONResponse(content= encoder)

@app.get("/koreksi-interval-r")
async def koreksi_interval_r():
    image = './images/koreksi.jpg'
    ecg = setup(file_path)
    peaks, _ = peaks_similarity(ecg.hart)
    rr, rr_corrected = group_rr_corrected(peaks)
    
    plt.figure(figsize=(20, 5))

    plt.title("Interval RR Dikoreksi", fontsize=20)
    plt.xlabel("Waktu (ms)", fontsize=20)
    plt.ylabel("RR-interval (ms)", fontsize=20)

    plt.plot(rr, color="red", linewidth=1, label="RR-interval")
    plt.plot(rr_corrected, color="green", linewidth=2, label="RR-interval setelah koreksi")
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=20)
    plt.savefig(image)
    plt.close()
     
    response = {
    'image': 'http://localhost:5000/images/koreksi.jpg',
    }
    
    encoder = jsonable_encoder(response)
    
    return JSONResponse(content= encoder)

@app.get("/timedomain")
async def timedomain():

    ecg = setup(file_path)
    peaks, _ = peaks_similarity(ecg.hart)
    _, rr_corrected = group_rr_corrected(peaks)
    
    ibi = "{:.2f}".format(np.mean(rr_corrected))
    sdnn ="{:.2f}".format(np.std(rr_corrected))
    rmssd ="{:.2f}".format(np.sqrt(np.mean(np.square(np.diff(rr_corrected)))))
    bpm ="{:.0f}".format(60000/np.mean(rr_corrected))
        
    response = {
        'ibi': float(ibi),
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'bpm': float(bpm)
    }
    
    encoder = jsonable_encoder(response)
    
    return JSONResponse(content= encoder)

@app.get("/frequencydomain")
async def frequencydomain():

    ecg = setup(file_path)
    peaks, _ = peaks_similarity(ecg.hart)
    _, rr_corrected = group_rr_corrected(peaks)
    

    # create interpolation function based on the rr-samples.
    fs = 4.0
    steps = 1 / fs

    # create interpolation function based on the rr-samples.
    x = np.cumsum(rr_corrected) / 1000.0
    f = interp1d(x, rr_corrected, kind='cubic')

    # now we can sample from interpolation function
    xx = np.arange(1, np.max(x), steps)
    rr_interpolated = f(xx)
    
        # Estimate the spectral density using Welch's method
    fxx, pxx = signal.welch(x=rr_interpolated, fs=fs, window='flattop')

    '''
    Segement found frequencies in the bands
     - Very Low Frequency (VLF): 0-0.04Hz
     - Low Frequency (LF): 0.04-0.15Hz
     - High Frequency (HF): 0.15-0.4Hz
    '''
    cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)

    # calculate power in each band by integrating the spectral density
    vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
    lf = trapz(pxx[cond_lf], fxx[cond_lf])
    hf = trapz(pxx[cond_hf], fxx[cond_hf])
    
    vlf_formated = "{:.2f}".format(vlf)
    lf_formated = "{:.2f}".format(lf)
    hf_formated = "{:.2f}".format(hf)

    response = {
        'vlf': float(vlf_formated),
        'lf': float(lf_formated),
        'hf': float(hf_formated)
    }
    
    encoder = jsonable_encoder(response)
    
    return JSONResponse(content= encoder)
    # return {"succeess"}

@app.get("/power-spektral")
async def power_spektral():
    image = './images/spektral.jpg'
    ecg = setup(file_path)
    peaks, _ = peaks_similarity(ecg.hart)
    _, rr_corrected = group_rr_corrected(peaks)
    
    fxx,pxx = fxx_pxx_interpolated(rr_corrected)
    
    
    plt.figure(figsize=(20, 5))
    plt.plot(fxx, pxx, color="k", linewidth=0.6)
    plt.title("Spektrum daya FFT", fontsize=20)

    # create interpolation function for plotting frequency bands
    psd_f = interp1d(fxx, pxx)

    # setup frequency bands for plotting
    x_vlf = np.linspace(0, 0.04, 100)
    x_lf = np.linspace(0.04, 0.15, 100)
    x_hf = np.linspace(0.15, 0.4, 100)

    plt.gca().fill_between(x_vlf, psd_f(x_vlf), color="white", edgecolor="black")
    plt.gca().fill_between(x_lf, psd_f(x_lf), color="white", edgecolor="black")
    plt.gca().fill_between(x_hf, psd_f(x_hf), color="white", edgecolor="black")

    # Menemukan nilai tengah untuk posisi label
    xh_label_center = sum(x_hf) / len(x_hf)
    yh_label_center = sum(psd_f(x_hf)) / len(psd_f(x_hf))

    # Menambahkan label di tengah grafik
    plt.annotate("HF", fontsize=16, xy=(xh_label_center, yh_label_center),
                xytext=(xh_label_center, yh_label_center),
                ha='center', va='center',
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Menemukan nilai tengah untuk posisi label
    xl_label_center = sum(x_lf) / len(x_lf)
    yl_label_center = sum(psd_f(x_lf)) / len(psd_f(x_lf))

    # Menambahkan label di tengah grafik
    plt.annotate("LF", fontsize=16, xy=(xl_label_center, yl_label_center),
                xytext=(xl_label_center, yl_label_center),
                ha='center', va='center',
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Menemukan nilai tengah untuk posisi label
    xvl_label_center = sum(x_vlf) / len(x_vlf)
    yvl_label_center = sum(psd_f(x_vlf)) / len(psd_f(x_vlf))

    # Menambahkan label di tengah grafik
    plt.annotate("VLF", fontsize=16, xy=(xvl_label_center, yvl_label_center),
                xytext=(xvl_label_center, yvl_label_center),
                ha='center', va='center',
                arrowprops=dict(facecolor='black', shrink=0.05))

    plt.gca().set_xlim(0, 0.5)
    plt.gca().set_ylim(0)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.xlabel("Frekuensi (Hz)", fontsize=16)
    plt.ylabel("PSD (ms^2/Hz)", fontsize=16)
    
    plt.savefig(image)
    plt.close()
    
    response = {
    'image': 'http://localhost:5000/images/spektral.jpg',
    }
    
    encoder = jsonable_encoder(response)
    
    return JSONResponse(content= encoder)

# run server

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5000)

# ========== fungsi ===========

def setup(data_path: any):
   
    # baca data dengan pandas
    df = pd.read_csv(data_path)
    
    # jadikan time sebagai index
    df_time_index = time_to_milliseconds(df)
    
    return df_time_index

def fxx_pxx_interpolated(rr_corrected):
        # sample rate for interpolation
    fs = 4.0
    steps = 1 / fs

    # create interpolation function based on the rr-samples.
    x = np.cumsum(rr_corrected) / 1000.0
    f = interp1d(x, rr_corrected, kind='cubic')

    # now we can sample from interpolation function
    xx = np.arange(1, np.max(x), steps)
    rr_interpolated = f(xx)
    
        # Estimate the spectral density using Welch's method
    fxx, pxx = signal.welch(x=rr_interpolated, fs=fs, window='flattop')
    return fxx,pxx

def peaks_similarity(df_time_index):
        # set range data


    # detect peaks
    peaks, similarity = detect_peaks(df_time_index, threshold=0.3)
    
   
    return peaks, similarity

def group_rr_corrected(peaks):
        # group peaks
    grouped_peaks = group_peaks(peaks)
    
    # RR-intervals are the differences between successive peaks
    rr = np.diff(grouped_peaks)
    
    rr_corrected = rr.copy()
    rr_corrected[np.abs(zscore(rr)) > 2] = np.median(rr)
    return rr, rr_corrected


def time_to_milliseconds(df):
    df["time"] = pd.to_datetime(df["time"])
    # Mendapatkan waktu awal dalam milidetik
    start_time = df["time"].min()

    # Mengubah waktu menjadi dalam milidetik dimulai dari 0
    df["time"] = (df["time"] - start_time).dt.total_seconds() * 1000
    df["time"] = df["time"].astype(int)
    df = df.set_index('time')
    return df

def detect_peaks(ecg_signal, threshold=0.3, qrs_filter=None):
    '''
    Peak detection algorithm using cross corrrelation and threshold
    '''
    if qrs_filter is None:
        # create default qrs filter
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)

    # normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    # calculate cross correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # return peaks (values in ms) using threshold
    return ecg_signal[similarity > threshold].index, similarity

def group_peaks(p, threshold=5):
    '''
    The peak detection algorithm finds multiple peaks for each QRS complex.
    Here we group collections of peaks that are very near (within threshold) and we take the median index
    '''
    # initialize output
    output = np.empty(0)

    # label groups of sample that belong to the same peak
    peak_groups, num_groups = label(np.diff(p) < threshold)

    # iterate through groups and take the mean as peak index
    for i in np.unique(peak_groups)[1:]:
        peak_group = p[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
    return output

