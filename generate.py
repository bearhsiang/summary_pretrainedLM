import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = '/hdd/gpt2_sum/model'

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

model.to(device)

max_len = 100
eos_ids = tokenizer.encode('<|endoftext|>') + [198]
sum_token = '<|isa|>'

while True:
    s = input('>>> ')
    if s == '!':
        break
    s = s + sum_token

    input_ids = torch.tensor(tokenizer.encode(s)).unsqueeze(0).to(device)

    past = None
    ids = input_ids

    while True:

        p, past = model(input_ids, past=past)
        input_ids = torch.argmax(p[:, -1, :], dim=-1).unsqueeze(-1)

        ids = torch.cat([ids, input_ids], dim=-1)
        if ids.shape[1] >= max_len:
            break

        if ids[0][-1].item() in eos_ids:
            break

    generated_ids = ids[0].detach().cpu()
    output = tokenizer.decode(generated_ids).strip()
    print(output)
