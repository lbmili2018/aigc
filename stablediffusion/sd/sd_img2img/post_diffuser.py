# -*- coding: utf-8 -*-
import requests
import json
import time

params={
    "img_path": "/root/limiao/orig.jpg",
    "strength": "0.5", # if strength=0, closed original image
}

url='http://0.0.0.0:8081/ai_universe'

time1=time.time()
html = requests.post(url, json.dumps(params))
print(html.text)
time2=time.time()
print('running time: ' + str(time2 - time1)[:5] + 's')



