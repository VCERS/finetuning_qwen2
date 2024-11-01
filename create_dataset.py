#!/usr/bin/python3

from absl import flags, app
from os import walk
from os.path import join, exists, splitext
from tqdm import tqdm
import json

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to input directory')
  flags.DEFINE_string('output', default = 'dataset.json', help = 'path to output dataset')

def main(unused_argv):
  system_message = """please label entities and relationships of the text in TANL(Text-to-Text Framework forMeta-Transfer Learning) format."""

  with open(FLAGS.input, 'r') as f:
    samples = json.loads(f.read())
  with open(FLAGS.output, 'w') as f:
    for sample in samples:
      input = sample['input']
      output = sample['output']
      messages = {"messages": [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': input},
        {'role': 'assistant', 'content': output}
      ]}
      f.write(json.dumps(messages, ensure_ascii = False) + '\n')

if __name__ == "__main__":
  add_options()
  app.run(main)

