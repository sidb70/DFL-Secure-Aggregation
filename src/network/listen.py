import http.server
import socketserver
import threading
import time
from typing import Callable
from logger import Logger
from urllib.parse import unquote
import json

## Global variable to store the callback and logger
msg_callback = None
logger = None

def set_globals(callback:Callable, log:Logger):
    global msg_callback
    global logger
    msg_callback = callback
    logger = log
class MsgHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
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
        global msg_callback
        global logger
        content_length = int(self.headers['Content-Length']) # Gets the size of data
        post_data = self.rfile.read(content_length) # Gets the data itself
        try:
           
            # post_data = unquote(post_data) # decode
            # post_data = post_data.split('&')
            # post_data = [x.split('=') for x in post_data]
            # post_data = {x[0]:x[1] for x in post_data} # convert to dictionary
            post_data = json.loads(post_data)
            #print(post_data)

            self.send_response(200) # Sends a response header
            self.end_headers()
            self.wfile.write(b'POST received') # Sends a response body
            
            if msg_callback is not None:
                msg_callback(post_data)
        except Exception as e:
            logger.log(f"Error: {e}")
            self.send_response(500)
            raise e


class MyServer(socketserver.TCPServer):
    allow_reuse_address = True
class ModelListener:
    def __init__(self, host:str, port: int):
        '''
        Initializes a Listen object with the specified host, port

        Args:
            host (str): The host address to bind the socket to.
            port (int): The port number to bind the socket to.
        '''

        self.host = host
        self.port = port

    def listen_for_models(self):
        """
        Listens for incoming models on the specified port and host.
        
        Args:
            timeout (int): The amount of time to listen for models (in seconds).
            port (int): The port number to listen on.
            host (str, optional): The host address to listen on. Defaults to 'localhost'.
            callback (Callable, optional): A callback function to be executed when a model is received. Defaults to None.
        """
        # Create the server
        self.server = MyServer((self.host, self.port), MsgHandler)
        # Run the server in a separate thread
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        logger.log("Listening on port "+ str(self.port))
    def shutdown(self):
        logger.log("Stop listening")
        # Shut down the server
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()