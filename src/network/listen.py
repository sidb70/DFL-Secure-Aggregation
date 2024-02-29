import http.server
import socketserver
import threading
import time
from typing import Callable
from logger import Logger

## Global variable to store the callback
model_callback = None
logger = None
class ModelHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # You can use self.custom_arg1 and self.custom_arg2 here
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        msg = f"Hello World"
        self.wfile.write(msg.encode('utf-8'))
    def do_POST(self):
        """
        Handle the HTTP POST request.

        This method reads the content of the request, decodes it from bytes to string,
        splits it into key-value pairs, converts it into a dictionary, and prints the result.
        Then it sends a response header with status code 200 and a response body of 'POST received'.
        If a model callback is defined, it calls the callback with the parsed data.

        """
        global model_callback
        global logger
        content_length = int(self.headers['Content-Length']) # Gets the size of data
        post_data = self.rfile.read(content_length) # Gets the data itself

        post_data = post_data.decode('utf-8') # Converts from bytes to string
        post_data = post_data.split('&')
        post_data = [x.split('=') for x in post_data]
        post_data = {x[0]:x[1] for x in post_data} # convert to dictionary
        #print(post_data)

        self.send_response(200) # Sends a response header
        self.end_headers()
        self.wfile.write(b'POST received') # Sends a response body
        logger.log("\nReceived model")
        
        if model_callback is not None:
            logger.log("Calling model callback\n")
            model_callback(post_data)


class MyServer(socketserver.TCPServer):
    allow_reuse_address = True

def listen_for_models(host:str ,port: int, timeout:int, log: Logger, callback:Callable=None):
    """
    Listens for incoming models on the specified port and host.
    
    Args:
        timeout (int): The amount of time to listen for models (in seconds).
        port (int): The port number to listen on.
        host (str, optional): The host address to listen on. Defaults to 'localhost'.
        callback (Callable, optional): A callback function to be executed when a model is received. Defaults to None.
    """
    global model_callback
    global logger
    model_callback = callback
    logger = log
    # Create the server
    server = MyServer((host, port), ModelHandler)
    # Run the server in a separate thread
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    logger.log("Listening on port "+ str(port))

    # Wait for x amount of time
    time.sleep(timeout) 
    
    logger.log("Stop listening")
    # Shut down the server
    server.shutdown()
    thread.join()
    server.server_close()