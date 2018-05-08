import time


def import_func(name):
    items = name.split('.')
    path = ".".join(items[:-1])
    func = items[-1]
    return getattr(__import__(path, fromlist=[func]), func)


class Logger:
    def __init__(self, logfile_path=""):
        if logfile_path != "":
            self.logfile = open(logfile_path, 'a')
        else:
            self.logfile = None

    def __del__(self):
        if self.logfile:
            self.logfile.close()

    def log(self, message):
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = " ".join([time_stamp, ">>>", message])
        print(message)

        if self.logfile:
            self.logfile.writelines([message + "\n"])
            self.logfile.flush()


class EvalInfo:
    """Stores the TER and Loss"""
    def __init__(self):
        self.sum_loss = 0.0
        self.sum_examples = 0.0
        self.sum_token_errors = 0.0
        self.sum_labels = 0.0

    def update(self, loss, examples, token_errors, labels):
        self.sum_loss += loss
        self.sum_examples += examples
        self.sum_token_errors += token_errors
        self.sum_labels += labels

    def avg_loss(self):
        return self.sum_loss / max(1.0, self.sum_examples)

    def avg_ter(self):
        return 100.0 * (self.sum_token_errors / max(1.0, self.sum_labels))

    def get_info(self, name="", epoch=0):
        info = "{name} Summary Epoch: [{epoch}]\tAverage Loss {loss:.3f}, Average TER {ter:.2f}\t".format(
            name=name, epoch=epoch+1, loss=self.avg_loss(), ter=self.avg_ter())
        return info
