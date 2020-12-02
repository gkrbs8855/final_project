# This Python file uses the following encoding: utf-8
from flask import Flask, render_template,request
import pickle
import json
import jsonify
import numpy as np
from PIL import Image
from tensorflow import keras
import pandas as pd
import csv
import time
app = Flask(__name__)

@app.route('/index')
def index():
    return render_template('kkkkkk/index.html')

@app.route('/self_check')
def self_check():
    return render_template('kkkkkk/self_check.html')

@app.route('/shop')
def shop():
    return render_template('kkkkkk/shop.html')

@app.route('/predict', methods=['POST'])
def predict():
    # my_res = flask.Response("차단되지롱")
    # my_res.headers["Access-Control-Allow-Origin"] = "*"
    # my_res.header("Access-Control-Allow-Origin", "*");
    global loaded_model
    global class_dict


    data=pd.read_csv('qqqqutf8.csv',delimiter='/t',encoding='CP949')
    print(data)
    time.sleep(0.5)

    image = Image.open(request.files['file'].stream)
    image = image.convert('RGB')  # 'L': greyscale, '1': 이진화, 'RGB' , 'RGBA', 'CMYK'
    image = image.resize((150, 150))

    image_numpy = np.array(image)
    x_test = image_numpy.reshape(1,150 , 150,3)
    # image_numpy = np.array(image_numpy)
    # image_numpy = image_numpy.reshape(-1,64)
    x_test = x_test / 255
    print(x_test.shape)
    print(x_test)

    pred = loaded_model.predict(x_test)
    print("예측결과:", pred)
    print("예측결과argmax:",  np.argmax(pred[0]))
    print("예측결과argmax:",  class_dict[np.argmax(pred[0])])

    data=''
    info=dict()
    classify=class_dict[np.argmax(pred[0])]
    if classify in info_dict:
        info = {'plant_name': info_dict[classify][0],
            'disease_name': info_dict[classify][1],
            'cause':info_dict[classify][2],
            'symptom':info_dict[classify][3],
            'care':info_dict[classify][4]}
        # data=jsonify(info)
    else:
        info = {'plant_name': classify,
                'disease_name': '정보수집중',
                'cause': '정보수집중',
                'symptom': '정보수집중',
                'care': '정보수집중'}
        # data = jsonify(info)

    return info#jsonify(info)

    # sepal_length = float(request.form['sepal_length'])
    # sepal_width = float(request.form['sepal_width'])
    # petal_length = float(request.form['petal_length'])
    # petal_width = float(request.form['petal_width'])
    #
    # list = [sepal_length,sepal_width,petal_length,petal_width]
    # myparam = np.array(list).reshape(1,-1)
    #
    # print(myparam.shape)  #(1, 4)
    #
    # loaded_model = pickle.load(open('mymodel.pkl', 'rb'))
    # result = loaded_model.predict(myparam)
    # print(result)
    #
    # return render_template('predict_result.html', MYRESULT=result)
#경북대 강원대 충남대 포지셔닝 어떤타겟을
if __name__ == '__main__':
    info_dict={'monstera__yellow':['몬스테라','몬스테라 노란잎','스트레스를 받거나 수분이 부족할수 있어요. 화분이 작을때, 뿌리들이 성장할 공간이 없어 양분 부족할수 있어요. 통풍이 안되어 물이 마르지 않아 과습일수도 있어요. 강한 직사광선에 노출되거나 냉해를 입었을 수 있어요. 단순 하엽일 수도 있어요.',	'잎이 노란색으로 변색되어요.',	'자주 식물을 건드리지 마세요. 물이 부족하면 수분 보충을 해주시고 줄기 잎 크기 고려하여 분갈이을 하거나 하지마세요. 4.통풍 과 창문 거리에 유의하세요. 항상 15~25도 온도 유지해주세요.'],
    'monstera__brown':['몬스테라','몬스테라 갈색반점',	'흙갈이나 과습으로 인한 뿌리손상이에요.',	'잎에 갈색반점 발생해요.','흙갈이 도중 뿌리손상이 의심되면 물을 적게 주면서 관리하세요. 큰 화분에 물을 주다 보면 물을 과하게 주게 되고 배수가 잘 안되면 너무 식물의 뿌리가 질식 및 부패해요.'],
    'Tomato___Bacterial_spot':['토마토','토마토 세균성 점무늬병',	"제대로 소독되지 않은 씨앗이 원인이예요. 병원균은 피해 잎이나 줄기에 붙어 월동하며 공기전염 , 고온다습한 환경의 시설재배 (하우스,베란다) 에서 많이 발생해요.",'잎에 검은빛 갈색 작은 점무늬가 나타나고 시간이 지나면서점무늬가 생긴 잎은 누렇게 변색되고 말라요.',	"환기를 철저히 하여 실내가 과습하지 않도록 조정하세요. 배수가 잘되는 흙에 완숙된 퇴비를 넣어서 키우세요. 급격한 온도변화가 없고 다습하지 않아해요. 병든 잎과 열매는 일찍 제거하세요. 병이 발생한 토양과 흙은 살균 처리하세요."],
    'Tomato___Septoria_leaf_spot':['토마토','토마토 흰 무늬병',	'여름철 장마기 이후 다습할 때 발생이 심해요. 병원균은 병든 부위에서 병자각의 형태로 겨울을 지낸 후, 병포자를 형성하여 공기전염 되요.	','감염부위에는 갈색 내지 암갈색의 작은 반점이 형성되고, 진전되면 병반의 내부는 회색으로 변해요. 그 곳에 흑색의 소립점이 보여요. 심하게 병든 잎은 황색으로 변하여 마르고, 일찍 떨어져요.',	'종자를 선별하고 소독하여 파종하세요. 재배 시 균형시비, 병든 잎과 열매는 일찍 제거하세요. 환기를 철저히 하여 실내가 과습하지 않도록 조정하세요'],
    'Tomato___Spider_mites Two-spotted_spider_mite':['토마토','토마토 점박이응애',	'작은 해충으로써 주로 잎의 뒷면에 서식하면서 흡즙해요. 연 9~10회 발생하며 4~5월에 주로 초본류(풀) 증식하고 7월경부터 나무로 이동해 8~9월에 밀도가 가장 높아요.', 	'잎이 약간 탈색된것 같이 보여요. 흰색 불규칙 반점과 같은 증상을 보이니 반드시 돋보기로 확인하세요.',	'7~9월에 피리다벤 수화제 1,000배액 또는 사이에노피라펜 액상수화제 2,000배액을 10일 간격으로 2회 이상 살포하세요.'],
    'Potato___Early_blight':['감자','감자 겹둥근무늬병',	'온도가 높고 비가 자주 오면 병의 발생이 많으며 비료의 부족은 병을 증가 시켜요',	'주로 잎에 많이 발생해요. 처음엔 작은 원형의 갈색 반점이 나타나며, 진전되면 대형 흑갈색 병반이 형성되요.',	'발생상습지는 돌려짓기를 하고 병든 잔사물은 제거하세요. 상처난 부분을 큐어링(치료)하여 괴경감염을 방지하세요.	'],
    'Potato___Late_blight':['감자','감자역병',	'병원균은 주로 병든 씨 감자에서 겨울을 지내므로 이것을 심으면 발병해요.',	'불규칙적인 누런색의 작은 점무늬가 생겨요. 이것이 차차 커지면서 갈색이 되고 흰 곰팡이가 생겨요.', 	"정기적으로 곰팡이 제거제를 뿌리며, 꼼꼼하게 성장 관리하세요. 역병의 발생의 많은 밭의 포장은 식물체의 잔해를 완전히 제거하세요."]
               }

    loaded_model = keras.models.load_model('RMSprop 병충해.h5')
    class_dict={'Apple___Apple_scab': 0,
                'Apple___Black_rot': 1,
                'Apple___Cedar_apple_rust': 2,
                'Apple___healthy': 3,
                'Blueberry___healthy': 4,
                'Cherry_(including_sour)___Powdery_mildew'  : 5,
                'Cherry_(including_sour)___healthy': 6,
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
                'Corn_(maize)___Common_rust_': 8,
                'Corn_(maize)___Northern_Leaf_Blight': 9,
                'Corn_(maize)___healthy': 10,
                'Grape___Black_rot': 11,
                'Grape___Esca_(Black_Measles)': 12,
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
                'Grape___healthy': 14,
                'Orange___Haunglongbing_(Citrus_greening)': 15,
                'Peach___Bacterial_spot': 16,
                'Peach___healthy': 17,
                'Pepper,_bell___Bacterial_spot': 18,
                'Pepper,_bell___healthy': 19,
                'Potato___Early_blight': 20,
                'Potato___Late_blight': 21,
                'Potato___healthy': 22,
                'Raspberry___healthy': 23,
                'Soybean___healthy': 24,
                'Squash___Powdery_mildew': 25,
                'Strawberry___Leaf_scorch': 26,
                'Strawberry___healthy': 27,
                'Tomato___Bacterial_spot': 28,
                'Tomato___Early_blight': 29,
                'Tomato___Late_blight': 30,
                'Tomato___Leaf_Mold': 31,
                'Tomato___Septoria_leaf_spot': 32,
                'Tomato___Spider_mites Two-spotted_spider_mite': 33,
                'Tomato___Target_Spot': 34,
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
                'Tomato___Tomato_mosaic_virus': 36,
                'Tomato___healthy': 37,
                'monstera__brown':38,
                'monstera__health':39,
                'monstera__yellow':40
                }
    class_dict = {v:k for k,v in class_dict.items()}
    app.debug = True
    app.run(port=9999)
