import listen
import threading
import requests
import time
import random
models=[]
def callback(data):
    global models
    models.append(data)
    print(models)
def test_handler():
    data = {"key1": random.random(), "key2": random.random()}
    response = requests.post('http://localhost:8000', data=data)
    print(response.content)
thread = threading.Thread(target=listen.listen_for_models, args=(30, 8000, 'localhost', callback))
time.sleep(1)
thread.start()
test_handler()

# if data shows up in the terminal, the test passes