import time
import torch
from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
import os

def generate_prompt_of_length(base_prompt, desired_token_len, tokenizer):
    """Extend base_prompt until it reaches approximately desired_token_len tokens"""
    tokens = tokenizer.encode(base_prompt, bos=True, eos=False)
    while len(tokens) < desired_token_len:
        tokens += tokenizer.encode(" " + base_prompt, bos=False, eos=False)
    trimmed_tokens = tokens[:desired_token_len]
    return tokenizer.decode(trimmed_tokens)

def generate_batch_prompts(base_prompt, input_len, batch_size, tokenizer):
    return [
        generate_prompt_of_length(base_prompt, input_len, tokenizer)
        for _ in range(batch_size)
    ]

def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def get_model_memory_mb(model):
    """Estimate the memory usage of the model in MB."""
    num_params = sum(p.numel() for p in model.parameters())
    param_size = 2  # Assuming float16 (2 bytes)
    return num_params * param_size / (1024 ** 2)

def bunc(batch_size,input_len,output_len,kv_caching):
    torch.manual_seed(1)
    for i in range(5):
        print(i+i*6)
        
def benchmark_inference(batch_size, input_len, output_len, kv_caching):
    torch.manual_seed(1)

    checkpoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
    model_path = os.path.join(checkpoint_dir, "consolidated.00.pth")

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()

    model_args.max_batch_size = batch_size
    model_args.max_seq_len = input_len + output_len
    model_args.kv_caching = kv_caching
    
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    model.to("cuda")
    model.eval()

    base_prompt = "Once upon a time in a galaxy far away"
    prompts = generate_batch_prompts(base_prompt, input_len, batch_size, tokenizer)

    start_time = time.time()

    # results = model.generate(
    #     tokenizer,
    #     prompts,
    #     max_gen_len=output_len,
    #     temperature=0.6,
    #     top_p=0.9,
    #     kv_caching=kv_caching,
    #     device="cuda",
    # )

    results = model.generate(
        tokenizer,
        prompts,
        max_gen_len=output_len,
        temperature=0.6,
        top_p=0.9,
        device="cuda",
    )

    elapsed = time.time() - start_time

    print(f"\nBatch size: {batch_size}")
    print(f"Input length: {input_len} tokens")
    print(f"Output length: {output_len} tokens")
    print(f"Inference time: {elapsed:.2f} seconds")
    print(f"Tokens per second: {batch_size * output_len / elapsed:.2f}")
    print(f"Model weights memory usage: {get_model_memory_mb(model):.2f} MB")
    print(f"Peak memory usage: {get_peak_memory_mb():.2f} MB")
    print("\n=== Sample Output ===")
    print(results[0]["generation"])
    
if __name__ == "__main__":
    benchmark_inference(batch_size=1, input_len=256, output_len=32, kv_caching=True)
