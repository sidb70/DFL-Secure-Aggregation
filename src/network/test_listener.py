import listen
models=[]
def callback(data):
    global models
    models.append(data)
    print(models)
listen.listen_for_models(timeout=30, host='localhost', port=8000, callback=callback)
