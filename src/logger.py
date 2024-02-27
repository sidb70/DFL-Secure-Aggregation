class Logger:
    def __init__(self, file_path):
        self.log_filename = file_path
        # if file exists, clear it
        with open(self.log_filename, "w") as f:
            f.write("")

    def log(self, message):
        with open(self.log_filename, "a") as f:
            f.write(message + "\n")
