class TodoStore:
    def __init__(self, path):
        self.path = path

    def add(self, title):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError

    def complete(self, title):
        raise NotImplementedError
