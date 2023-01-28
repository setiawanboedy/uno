from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import matplotlib.pyplot as plt
import os
import uvicorn
import base64
import heartpy as hb


app = FastAPI()

app.mount('/data', StaticFiles(directory="data"), name="data")


def setup(data: any):
    ecg = hb.get_data(data, column_name='hart')
    ecg_time = hb.get_data(data, column_name='time')
    sample_rate = round(hb.get_samplerate_datetime(ecg_time, timeformat='%Y-%m-%d %H:%M:%S.%f'))
    
    filtered = hb.filter_signal(ecg, cutoff = 0.05, sample_rate = sample_rate, filtertype='notch')
    
    wd, m = hb.process(hb.scale_data(filtered), sample_rate, calc_freq=True, freq_method='fft')
    return ecg, wd, m

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
    ecg, wd, m = setup(file_path)
    
    original = hb.scale_data(ecg[0:1000])
    
    ori = [o for o in original]
    
    response = {
        'ecg': ori
    }
    
    encoder = jsonable_encoder(response)
    
    return JSONResponse(content= encoder)

@app.get("/data")
async def original_signal():
    file_path = "./data/heart.csv"
    ecg, wd, m = setup(file_path)
    
    response = {
        'ibi': m['ibi'],
        'sdnn': m['sdnn'],
        'sdsd': m['sdsd'],
        'rmssd': m['rmssd'],
        'bpm': m['bpm'],
        'lf': m['lf'],
        'hf': m['hf']
    }
    
    encoder = jsonable_encoder({'data': response})
    
    return JSONResponse(content= encoder)
    

@app.get("/spectrum")
async def spectrum():
    file_path = "./data/heart.csv"
    ecg, wd, m = setup(file_path)
    
    file_image = "./images/spectrum.png"
    
    os.remove(file_image)
    plt.title("Frekuensi Spektrum Sinyal Jantung")
    plt.xlim(0,0.6)
    plt.ylim(0, 500)
    plt.plot(wd['frq'], abs(wd['psd']))
    plt.xlabel("Frekuensi (Hz)")
    plt.savefig(file_image)
    plt.close()
    
    
    with open(file_image, "rb") as image_file:
        encode_image = base64.b64encode(image_file.read())
    return encode_image
        

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5000)