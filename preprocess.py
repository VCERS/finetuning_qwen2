#!/usr/bin/python3

from shutil import rmtree
from os import mkdir
from os.path import exists
from absl import flags, app
import pandas as pd
from models import Qwen2
from chains import example_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to csv')
  flags.DEFINE_string('output_dir', default = 'output', help = 'path to directory of output')
  flags.DEFINE_integer('size', default = 150, help = 'dataset number')
  flags.DEFINE_integer('pad', default = 100, help = 'padding token number')
  flags.DEFINE_boolean('locally', default = False, help = 'whether to run the model locally')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  tokenizer, llm = Qwen2(FLAGS.locally)
  example_chain_ = example_chain(llm, tokenizer)
  df = pd.read_csv(FLAGS.input_csv)
  for idx in range(FLAGS.size):
    description = df.iloc[idx]['Description']
    example = example_chain_.invoke({'patent': description})
    text_tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(description)
    example_tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenizer_str(example)

if __name__ == "__main__":
  add_options()
  app.run(main)

