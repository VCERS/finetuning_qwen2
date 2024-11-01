#!/usr/bin/python3

from os import environ
from torch import device
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, \
                TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.llms.base import LLM

def Customized(locally = True, ckpt = None):
  assert locally == True, "customized model can only run locally!"
  class Customized(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    def __init__(self,):
      super().__init__()
      self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code = True)
      self.model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code = True)
      self.model = self.model.to(device('cuda'))
      self.model.eval()
    def _call(self, prompt, stop = None, run_manager = None, **kwargs):
      logits_processor = LogitsProcessorList()
      logits_processor.append(TemperatureLogitsWarper(0.8))
      logits_processor.append(TopPLogitsWarper(0.8))
      inputs = self.tokenizer(prompt, return_tensors = 'pt')
      inputs = inputs.to(device('cuda'))
      outputs = self.model.generate(**inputs, logits_processor = logits_processor, use_cache = True, do_sample = True, max_length = 131072)
      outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
      response = self.tokenizer.decode(outputs)
      return response
    @property
    def _llm_type(self):
      return "customized"
  llm = Customized()
  return llm.tokenizer, llm

