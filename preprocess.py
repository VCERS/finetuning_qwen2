#!/usr/bin/python3

from shutil import rmtree
from os import mkdir
from os.path import exists, join
from absl import flags, app
import pandas as pd
from fuzzywuzzy import fuzz
from models import Qwen2
from chains import example_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_json', default = None, help = 'path to csv')
  flags.DEFINE_string('output_dir', default = 'output', help = 'path to directory of output')
  flags.DEFINE_integer('size', default = 150, help = 'dataset number')
  flags.DEFINE_integer('pad', default = 100, help = 'padding token number')
  flags.DEFINE_boolean('locally', default = False, help = 'whether to run the model locally')

def find_near_matches(long_string, sub_string, threshold=80):
    matches = []
    for i in range(len(long_string) - len(sub_string) + 1):
        window = long_string[i:i+len(sub_string)]
        similarity = fuzz.ratio(window, sub_string)
        if similarity >= threshold:
            matches.append((i, window, similarity))
    return matches

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  tokenizer, llm = Qwen2(FLAGS.locally)
  example_chain_ = example_chain(llm, tokenizer)
  df = pd.read_json(FLAGS.input_json)
  for idx in range(FLAGS.size):
    description = df.iloc[idx]['Description']
    example = example_chain_.invoke({'patent': description})
    example = example[example.find('\n\n') + 2:]
    matches = find_near_matches(description, example, 80)
    if len(matches) == 0: continue
    start = matches[0][0]
    end = matches[0][0] + len(matches[0][1])
    text = description[start - FLAGS.pad:end + FLAGS.pad]
    with open(join(FLAGS.output_dir, '%d.txt' % idx),'w') as f:
      f.write(text)

if __name__ == "__main__":
  add_options()
  app.run(main)

