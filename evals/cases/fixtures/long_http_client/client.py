class RetryableError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, sender, retries=2):
        self.sender = sender
        self.retries = retries

    def request(self, url):
        return self.sender(url)
