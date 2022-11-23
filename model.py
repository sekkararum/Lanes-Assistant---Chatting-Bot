import pickle

# global variable
global model_sbmptn, scaler_sbmptn

def load_sbmptn():
    global model_sbmptn, scaler_sbmptn
    model_sbmptn = pickle.load(open('model/sbmptn/model_ds.pkl', 'rb'))
    scaler_sbmptn = pickle.load(open('model/sbmptn/scaler_ds.pkl', 'rb'))

def prediksi_sbmptn(data):
    data = scaler_sbmptn.transform(data)
    prediksi = int(model_sbmptn.predict(data))

    if prediksi == 0:
        hasil_prediksi = "lolos!"
    else:
        hasil_prediksi = "tidak lolos!"
    return hasil_prediksi

# global variable
global model_snmptn, scaler_snmptn

def load_snmptn():
    global model_snmptn, scaler_snmptn
    model_snmptn = pickle.load(open('model/snmptn/model_ds.pkl', 'rb'))
    scaler_snmptn = pickle.load(open('model/snmptn/scaler_ds.pkl', 'rb'))

def prediksi_snmptn(data):
    data = scaler_snmptn.transform(data)
    prediksi = int(model_snmptn.predict(data))

    if prediksi == 0:
        hasil_prediksi = "lolos!"
    else:
        hasil_prediksi = "tidak lolos!"
    return hasil_prediksi