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
  system_message = """please label entities and relationships of the text as following examples.

examples:

example1:

input:
The Baeyer-Villiger oxidation of ketones with Oxone(r) in the presence of ionic liquids as solvents Oxone(r) (4.0 mmol) was added to a solution of ketone (4.0 mmol) in ionic liquid (3.0 g) and stirred at 40 degC for 2.5-20 h (depending on the reaction rate).

output:
The Baeyer-Villiger oxidation of [ ketones | Material ] with Oxone(r) in the presence of ionic liquids as solvents [ Oxone(r) | Material | Recipe Precursor of = added ] [ (4.0 | Number | Number Of = mmol ] [ mmol) | Amount-Unit | Amount Of = Oxone(r) ] was [ added | Operation | Next Operation = stirred | | = Operation ] to a [ solution | Material-Descriptor | Descriptor Of = ketone ] of [ ketone | Material | Recipe Precursor of = added ] [ (4.0 | Number | Number Of = mmol ] [ mmol) | Amount-Unit | Amount Of = ketone ] in [ ionic liquid | Material | Solvent Material of = added ] [ (3.0 | Number | Number Of = g ] [ g) | Amount-Unit | Amount Of = ionic liquid ] and [ stirred | Operation | Next Operation = monitored | | = Operation ] at [ 40 | Number | Number Of = degC ] [ degC | Condition-Unit | Condition Of = stirred ] for [ 2.5-20 | Number | Number Of = h ] [ h | Condition-Unit | Condition Of = stirred ] (depending on the [ reaction rate). | Condition-Type ]

example2:

input:
The progress of the reaction was monitored by GC or HPLC. After this time, the post reaction mixture was dissolved in CH2Cl2 and filtered.

output:
The progress of the reaction was [ monitored | Operation | Next Operation = dissolved | | = Operation ] by [ GC | Characterization-Apparatus | Apparatus Of = monitored ] or [ HPLC. | Characterization-Apparatus | Apparatus Of = monitored ] After this time, the post reaction [ mixture | Material ] was [ dissolved | Operation | Next Operation = filtered | | = Operation ] in [ CH2Cl2 | Material | Solvent Material of = dissolved ] and [ filtered. | Operation | Next Operation = concentrated | | = Operation ]
"""

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

