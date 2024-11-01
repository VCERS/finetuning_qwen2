#!/usr/bin/python3

from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.output_parsers.regex import RegexParser

def label_tanl_template(tokenizer):
  messages = [
    {'role': 'system', 'content': 'please label entities and relationships of the text in TANL(Text-to-Text Framework forMeta-Transfer Learning) format.'},
    {'role': 'user', 'content': '{text}'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['text'])
  return template
