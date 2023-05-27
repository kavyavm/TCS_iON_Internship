# importing necessary libraries and functions
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__) #Initialize the flask App

model = pickle.load(open("model.pkl", "rb")) # loading the trained model

minmax = pickle.load(open("scaling_features.pkl","rb")) #loading the scaled data


@app.route('/') # Homepage
def home():
    return render_template('home.html')

@app.route('/former') # Homepage
def former():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():

    battery_power = request.form['battery_power']
    blue = request.form['blue']
    if (blue=='Yes'):
        blue=1
    else:
        blue=0
    clock_speed = request.form['clock_speed']
    dual_sim = request.form['dual_sim']
    if (dual_sim=='Yes'):
        dual_sim=1
    else: 
        dual_sim=0
    fc = request.form['fc']
    four_g = request.form['four_g']
    if (four_g =='Yes'):
        four_g = 1
    else: 
        four_g = 0
    int_memory = request.form['int_memory']
    m_dep = request.form['m_dep']
    mobile_wt  =request.form['mobile_wt']
    n_cores = request.form['n_cores']
    pc = request.form['pc']
    px_height = request.form['px_height']
    px_width = request.form['px_width']
    ram = request.form['ram']
    sc_h = request.form['sc_h']
    sc_w = request.form['sc_w']
    talk_time = request.form['talk_time']
    three_g = request.form['three_g']
    if (three_g =='Yes'):
        three_g = 1
    else: 
        three_g = 0
    touch_screen = request.form['touch_screen']
    if (touch_screen =='Yes'):
        touch_screen = 1 
    else: 
        touch_screen = 0
    wifi=request.form['wifi']
    if (wifi =='Yes'):
        wifi = 1
    else: 
        wifi = 0


    df = {'battery_power':[battery_power],'blue':[blue],'clock_speed':[clock_speed],'dual_sim':[dual_sim],'fc':[fc],
        'four_g':[four_g],'int_memory':[int_memory],'m_dep':[m_dep],'mobile_wt':[mobile_wt],'n_cores':[n_cores],'pc':[pc],
        'px_height':[px_height],'px_width':[px_width],'ram':[ram],'sc_h':[sc_h],'sc_w':[sc_w],'talk_time':[talk_time],
        'three_g':[three_g],'touch_screen':[touch_screen],'wifi':[wifi]}

   
    data=pd.DataFrame(data=df)
        
    print(data)
    input_data= minmax.transform(data)

    prediction=model.predict(input_data)
    pred =prediction[0]
    output='Error'
    if pred==0:
        output='Your expected price range for this smartphone is: Low Cost'
    elif pred==1:
        output='Your expected price range for this smartphone is: Medium Cost'
    elif pred==2:
        output='Your expected price range for this smartphone is: High Cost'
    else :
        output='Your expected price range for this smartphone is:Very High Cost'

    coefficients = model.coef_

    avg_importance =np.mean(np.abs(coefficients),axis=0)
    feature_importance = pd.DataFrame({'Feature': data.columns, 'Importance': avg_importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance['Rank']= feature_importance['Importance'].rank(ascending=False)

    
    feature_importance.plot.bar(x='Feature', figsize=(8,5),fontsize=10)
    plt.savefig('static/images/plot.png')
        
    return render_template('index.html',results=output,url='/static/images/plot.png', tables=[feature_importance.to_html(classes='data fl-table',header="true", index=False)])
                                                                    


if __name__ == "__main__":
    app.run(debug=True)