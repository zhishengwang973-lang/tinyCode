import json
import tempfile
import unittest
from pathlib import Path
from todo import TodoStore


class TodoTests(unittest.TestCase):
    def test_add_complete_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "todos.json"
            store = TodoStore(path)
            store.add("write tests")
            self.assertEqual([{"title": "write tests", "done": False}], store.list())
            store.complete("write tests")
            self.assertTrue(TodoStore(path).list()[0]["done"])
            self.assertEqual(json.loads(path.read_text())[0]["title"], "write tests")

    def test_invalid_json_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "todos.json"
            path.write_text("not json")
            self.assertEqual([], TodoStore(path).list())
