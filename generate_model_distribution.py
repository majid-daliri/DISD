import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text and token probabilities across different models.")
    parser.add_argument("--model_large", type=str, required=True, help="Name of the large model")
    parser.add_argument("--model_medium", type=str, required=True, help="Name of the medium model")
    parser.add_argument("--model_small", type=str, required=True, help="Name of the small model")
    parser.add_argument("--input_text", type=str, required=True, help="Input text to generate from")
    parser.add_argument("--num_tokens", type=int, default=32, help="Number of tokens to generate")
    return parser.parse_args()


def sanitize_filename(text):
    return text.replace(" ", "_").replace("?", "")


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def generate_text_distribution(args):
    device = 'cuda'

    tokenizer = AutoTokenizer.from_pretrained(args.model_large)
    model_large = AutoModelForCausalLM.from_pretrained(args.model_large, device_map="auto", torch_dtype=torch.bfloat16)
    model_medium = AutoModelForCausalLM.from_pretrained(args.model_medium, device_map="auto",
                                                        torch_dtype=torch.bfloat16)
    model_small = AutoModelForCausalLM.from_pretrained(args.model_small, device_map="auto", torch_dtype=torch.bfloat16)

    input_ids = tokenizer(args.input_text, return_tensors="pt").to(device).input_ids
    generated_tokens = []

    df_large = pd.DataFrame()
    df_medium = pd.DataFrame()
    df_small = pd.DataFrame()

    for _ in range(args.num_tokens):
        with torch.no_grad():
            next_token_id, probs_large = generate(model_large, input_ids)
            next_token_id_medium, probs_medium = generate(model_medium, input_ids)
            next_token_id_small, probs_small = generate(model_small, input_ids)

            df_large = log_probs(df_large, probs_large)
            df_medium = log_probs(df_medium, probs_medium)
            df_small = log_probs(df_small, probs_small)

            generated_tokens.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

    base_path = f"./data/{sanitize_filename(args.input_text)}"
    save_distribution(base_path, args.model_large, df_large)
    save_distribution(base_path, args.model_medium, df_medium)
    save_distribution(base_path, args.model_small, df_small)

    generated_text = tokenizer.decode(generated_tokens)
    print("Generated text:", generated_text)


def log_probs(df, probs):
    if df.empty:
        df = pd.DataFrame(columns=[f'prob_{j}' for j in range(probs.shape[0])])
    df.loc[len(df)] = probs.cpu().numpy().flatten()
    return df


def save_distribution(base_path, model_name, df):
    model_name_dict = model_name.replace("/", "-")
    ensure_dir(f"{base_path}/{model_name_dict}")
    df.to_csv(f"{base_path}/{model_name_dict}_probs.csv", index=False)


def generate(model, input_ids):
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1).item()
    return next_token_id, probs.squeeze(0)


if __name__ == "__main__":
    args = parse_arguments()
    generate_text_distribution(args)