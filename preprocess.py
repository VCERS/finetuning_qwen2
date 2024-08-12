#!/usr/bin/python3

from shutil import rmtree
from absl import flags, app
import pandas as pd
from models import Qwen2
from chains import example_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to csv')
  flags.DEFINE_string('output_dir', default = 'output', help = 'path to directory of output')
  flags.DEFINE_integer('size', default = 150, help = 'dataset number')
  flags.DEFINE_boolean('locally', default = False, help = 'whether to run the model locally')

def main(unused_argv):
  tokenizer, llm = Qwen2(FLAGS.locally)
  example_chain_ = example_chain(llm, tokenizer)
  df = pd.read_csv(FLAGS.input_csv)
  for idx in range(FLAGS.size):
    description = df.iloc[idx]['Description']
    example = example_chain_.invoke({'patent': description})

if __name__ == "__main__":
  add_options()
  app.run(main)

