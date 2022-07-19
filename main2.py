import torch
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

class Input(BaseModel):
    text: str = Field(
        title = '문장을 입력해주세요',
        max_length = 128
    )

    max_length: int = Field(
        128,
        ge=5,  #greater than equal to 5
        le=128 #less than equal to 128
    )

    repetition_penalty: float = Field(
        2.0,
        ge=0.0,
        le=2.0
    )

class Output(BaseModel):
    generated_text: str

def generate_text(input: Input) -> Output:
    input_ids = tokenizer.encode(input.text, return_tensors='pt')
    gen_ids = model.generate(input_ids,
                            max_length=input.max_length,
                            repetition_penalty=input.repetition_penalty,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            use_cache=True)
    generated = tokenizer.decode(gen_ids[0])
    
    return Output(generated_text = generated)

    #Terminal

    #pip install streamlit==1.3
    #pip uninstall click-utils
    #pip install click==8.0

    #opyrator launch-ui main2:generate_text
