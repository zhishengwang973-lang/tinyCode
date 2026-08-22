import unittest

from tinyCode.security.command_classifier import is_read_only_command


class CommandClassifierTests(unittest.TestCase):
    def test_known_read_only_commands_and_pipelines_are_allowed(self):
        allowed = [
            "pwd",
            "/bin/ls -la",
            "rg TODO tinyCode | head -20",
            'find . -name "*.py" -type f',
            "git diff --stat && git status --short",
            "sed -n 1,20p README.md",
            "ls missing 2>/dev/null",
            "find src -type f 2>>/dev/null | head -20",
            "printf status >/dev/null && pwd",
        ]
        for command in allowed:
            with self.subTest(command=command):
                self.assertTrue(is_read_only_command(command))

    def test_side_effects_and_unknown_commands_are_rejected(self):
        rejected = [
            "",
            "rm file.txt",
            "ls > files.txt",
            "ls 2> errors.txt",
            "ls >/tmp/files.txt",
            "ls 2>&1",
            "cat < secret.txt",
            "echo `touch result.txt`",
            "echo $(touch result.txt)",
            "find . -delete",
            "find . -exec touch {} ;",
            "sed -i.bak s/a/b/ file.txt",
            "git reset --hard",
            "rg TODO | xargs rm",
        ]
        for command in rejected:
            with self.subTest(command=command):
                self.assertFalse(is_read_only_command(command))


if __name__ == "__main__":
    unittest.main()
