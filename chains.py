#!/usr/bin/python3

from prompts import label_tanl_template

def label_tanl_chain(llm, tokenizer):
  tanl_template = extract_tanl_template(tokenizer)
  tanl_chain = extract_tanl_template | llm
  return tanl_chain

