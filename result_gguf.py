import time
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import csv
from llama_cpp import Llama, LlamaTokenizer # LlamaTokenizer might not be explicitly needed if model handles it

#####################################################################
# === SPEC NOTICE ===
# This script is an ADAPTATION.
# PPL calculation, timing, and throughput logic are REIMPLEMENTED
# to work with llama-cpp-python's API.
# The core goal (measuring PPL and throughput) is maintained.
#####################################################################

# === (Optional) Custom generate equivalent for llama-cpp-python ===
# For simplicity, we'll use model.create_completion or model.generate later.
# A custom loop would involve model.eval() and manual sampling.

def evaluate_ppl_llamacpp(model: Llama, tokenizer_for_eval, dataset_name: str, dataset_config: str, split: str, seq_len: int):
    """
    Evaluates perplexity for a Llama.cpp model.
    Note: This PPL calculation is adapted and  lla  might differ slightly from
          HF Transformers' exact nn.CrossEntropyLoss behavior internallybash 

          but aims for the same conceptual goal.
    tokenizer_for_eval: While Llama object has a tokenizer, for datasets,
                       it might be useful to pass an HF one if GGUF is based on it,
                       or rely on model.tokenize() carefully. For simplicity, we'll
                       use model.tokenize here.
    seq_len: The sequence length to process for PPL, typically model.n_ctx().
    """
    test_dataset = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(test_dataset["text"])

    # Tokenize the entire dataset
    # llama-cpp-python's tokenize usually expects bytes
    tokens = model.tokenize(text.encode("utf-8"), add_bos=False)

    nlls = []
    processed_tokens = 0

    # Non-overlapping windows for PPL, similar to original script
    num_batches = len(tokens) // seq_len

    for i in tqdm(range(num_batches), desc="Evaluating PPL with Llama.cpp..."):
        batch_start = i * seq_len
        batch_end = (i + 1) * seq_len
        batch = tokens[batch_start:batch_end]

        if len(batch) < 2: # Need at least two tokens to predict something
            continue

        # Reset model state for each independent batch
        model.reset()
        
        # Evaluate the batch. model.scores will be populated.
        # The `eval` method processes tokens and updates internal state.
        # The `scores` attribute should then hold the logits for all positions.
        model.eval(batch)
        
        # model.scores is a flat list (or CFFI array) of logits: (n_tokens_evaluated * n_vocab)
        # It contains logits for predicting token N+1 given tokens 0...N.
        all_logits_np = np.array(model.scores).reshape(-1, model.n_vocab())

        # We want logits for predicting batch[1:] from context batch[0:-1]
        shift_logits = all_logits_np[:-1, :] # Logits for all but the last token's prediction
        shift_labels = np.array(batch[1:])   # Actual next tokens to predict

        # Clip shift_labels to be within the valid vocabulary range
        # This is a safeguard. Ideally, tokenization should not produce out-of-vocab IDs
        # if the tokenizer is correctly aligned with the model's vocabulary.
        vocab_size = model.n_vocab()
        shift_labels_clipped = np.clip(shift_labels, 0, vocab_size - 1)

        # Calculate Cross-Entropy manually
        # log_softmax(x) = x - log(sum(exp(x)))
        log_probs = shift_logits - np.log(np.sum(np.exp(shift_logits), axis=-1, keepdims=True))
        
        # Get the log probability of the actual next token
        neg_log_likelihood_batch = -log_probs[np.arange(len(shift_labels)), shift_labels]

        # Get the log probability of the actual next token using clipped labels
        # neg_log_likelihood_batch = -log_probs[np.arange(len(shift_labels_clipped)), shift_labels_clipped]
        
        nlls.extend(neg_log_likelihood_batch)
        processed_tokens += len(shift_labels)

    if processed_tokens == 0:
        return float('inf')

    mean_nll = np.sum(nlls) / processed_tokens
    ppl = np.exp(mean_nll)
    return ppl.item()


def main():
    ############################## Set Up ##############################
    np.random.seed(0) # For numpy if used in sampling, llama-cpp has its own seed for generation
    random.seed(0)

    max_new_tokens = 256  # Number of new tokens to generate
    
    ### === TODO: Load your model (llama-cpp-python) ===
    # gguf_file = "hankleetw/llama_3.2_3b_instruct/llama-3.2-3B-instruct_q8_0.gguf" # Path to your GGUF file
    # For llama-cpp-python, 'model_name' is just for reference, path is key.
    # model_name_ref = "hankleetw/llama_3.2_3b_instruct" 

    # n_gpu_layers: Number of layers to offload to GPU. -1 for all, 0 for CPU only.
    # Adjust n_ctx (context size) as needed, usually matches what the model was trained for.
    # Set verbose=False to reduce llama.cpp output, True for debugging.
    # logits_all=True in the constructor is for some internal generation methods,
    # for PPL via `eval()`, `model.scores` is the key.
    # print(f"Loading GGUF model from: {gguf_file}")
    try:
        # model = Llama(
        #     model_path=gguf_file,
        #     n_ctx=2048, # Context window size
        #     n_gpu_layers=-1, # Offload all possible layers to GPU
        #     logits_all=True,
        #     verbose=False
        # )
        model = Llama.from_pretrained(
            repo_id="hankleetw/llama32_lora_merged",
            filename="*q4_k_m.gguf",
            n_ctx=2048, # Context window size
            n_gpu_layers=-1, # Offload all possible layers to GPU
            logits_all=True,
            split_mode=0,
            flash_attn=True, # Enable flash attention for better performance
            verbose=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have the correct GGUF file path and llama-cpp-python is correctly installed with GPU support if n_gpu_layers > 0.")
        return
    
    # Tokenizer is integrated into the Llama object.
    # For specific HF tokenizer use, it's more complex and depends on GGUF compatibility.
    # We'll rely on model.tokenize and model.detokenize.
    
    print("Model loaded successfully.")

    ####################################################################
    
    warmup_prompt = "Explain what AI is."
    print("Starting warm-up...")
    for i in tqdm(range(3), desc="Warm Up..."): # Fewer warm-ups for faster script run
        _ = model.create_completion(
            warmup_prompt,
            max_tokens=max_new_tokens,
            temperature=0.1 # Low temperature for more deterministic warmup
        )
    print("Warm-up complete.")
        
    prompt = "How to learn a new language?"
    tputs = []
    time_record = []

    print("Starting test inference...")
    for _ in tqdm(range(10), desc="Test Inference"):
        start_time = time.perf_counter()

        completion = model.create_completion(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.7, # Typical temperature for generation
            echo=False # Don't include prompt in the output text
        )
        
        end_time = time.perf_counter()

        elapsed_s = end_time - start_time
        time_record.append(elapsed_s)

        generated_text_only = completion['choices'][0]['text']
        # num_generated_tokens = len(model.tokenize(generated_text_only.encode("utf-8"), add_bos=False)) # Alternative
        num_generated_tokens = completion['usage']['completion_tokens']


        if elapsed_s > 0:
            tput = num_generated_tokens / elapsed_s
            tputs.append(tput)
        else:
            tputs.append(float('inf')) # Avoid division by zero
            
    # Get one response for printing
    final_completion = model.create_completion(prompt, max_tokens=max_new_tokens, echo=False, temperature=0.7)
    response = final_completion['choices'][0]['text']
    
    # Remove outliers for throughput calculation (middle 60%)
    if len(tputs) > 4:
        sorted_tputs = np.sort(tputs)[int(len(tputs)*0.2):int(len(tputs)*0.8)] # More robust outlier removal
        org_tput = np.mean(sorted_tputs) if len(sorted_tputs) > 0 else np.mean(tputs)
    elif len(tputs) > 0:
        org_tput = np.mean(tputs)
    else:
        org_tput = 0.0

    print(f'\nPrompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record (seconds): {time_record}')
    print(f'Throughput Record (toks/s): {tputs}\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput:.2f} toks/s')

    # PPL Evaluation
    # Using model.n_ctx() for seq_len in PPL.
    # Ensure the dataset is available.
    print("Starting PPL evaluation...")
    ppl = evaluate_ppl_llamacpp(
        model,
        tokenizer_for_eval=None, # Using model.tokenize
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test",
        seq_len=model.n_ctx() # Or a smaller value like 512, 1024 if n_ctx is too large for practical PPL
    )
    print(f"Perplexity (PPL) on wikitext-2-raw-v1/test: {ppl:.4f}")
    
    # Save results to CSV
    rounded_tput = round(org_tput, 1)
    ppl_rounded = round(ppl, 2)

    with open("result_llamacpp.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl_rounded])
        writer.writerow([1, rounded_tput])
    print(f"Results saved to result_llamacpp.csv")

if __name__ == '__main__':
    main()