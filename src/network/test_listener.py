import listen
import threading
import requests
import time
import random
models=[]
def callback(data):
    global models
    models.append(data)
def test_handler():
    data = {"key1": random.random(), "key2": random.random()}
    response = requests.post('http://localhost:8000', data=data)
    print(response.content)
thread = threading.Thread(target=listen.listen_for_models, args=(2, 8000, 'localhost', callback))
time.sleep(1)
thread.start()
for i in range(3):
    test_handler()
print(models)
assert len(models) == 3
# if data shows up in the terminal, the test passes