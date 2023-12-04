from transformers import T5ForConditionalGeneration, T5Tokenizer
import typing
finetuned_model_path = 't5_finetuned'

class Ollama:
  finetuned_model: typing.Any
  tokenizer: typing.Any
  max_length: int = 10000
  
  def __init__(self) -> None:
    self.finetuned_model = T5ForConditionalGeneration.from_pretrained("t5_finetuned")
    self.tokenizer = T5Tokenizer.from_pretrained(finetuned_model_path)
    self.finetuned_model.eval()
  
  def generateResp(self, query: str) -> str:
    tokenized_query = self.tokenizer(query, return_tensors="pt").input_ids
    output_ids = self.finetuned_model.generate(tokenized_query, max_length=self.max_length)
    return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
  
if __name__ == "__main__":
  x = Ollama()
  print(x.generateResp("to office from home"))
