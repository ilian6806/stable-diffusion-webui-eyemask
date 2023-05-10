import os
import sys
import re
import random

from .constants import script_wildcards_dir


class WildcardsGenerator():

    def __init__(self):
        self.wildcards_warned_about_files = {}
        self.wildcard_indexes = {}

    def build_prompt(self, prompt):
        if re.search("_{2}[a-z_]+_{2}", prompt):
            return "".join(self.replace_wildcard(chunk) for chunk in prompt.split("__"))
        return prompt

    def get_index(self, file_name, max_index):
        if not file_name in self.wildcard_indexes or self.wildcard_indexes[file_name] == max_index:
            self.wildcard_indexes[file_name] = 0
        else:
            self.wildcard_indexes[file_name] += 1
        return self.wildcard_indexes[file_name]

    def replace_wildcard(self, text):

        if " " in text or len(text) == 0:
            return text

        replacement_file = os.path.join(script_wildcards_dir, f"{text}.txt")
        if os.path.exists(replacement_file):
            with open(replacement_file, encoding="utf8") as f:
                lines = f.read().splitlines()
                if text[-5:] == '_each':
                    return lines[self.get_index(text, len(lines) - 1)]
                else:
                    return random.Random().choice(lines)
        else:
            if replacement_file not in self.wildcards_warned_about_files:
                print(f"File {replacement_file} not found for the __{text}__ wildcard.", file=sys.stderr)
                self.wildcards_warned_about_files[replacement_file] = 1

        return text
