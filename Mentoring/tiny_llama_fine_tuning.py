import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import Dataset, DataLoader

# Load pre-trained model and tokenizer
model_name = "distilgpt2"  # Example: Using a smaller model for demonstration
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.train()

# Create the dataset class
class ControlsDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = data['Input']
        self.outputs = data['Output']
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]

        input_tokens = self.tokenizer.encode(input_text, truncation=True, max_length=self.max_length)
        output_tokens = self.tokenizer.encode(output_text, truncation=True, max_length=self.max_length)

        input_ids = input_tokens + [self.tokenizer.eos_token_id]
        output_ids = output_tokens + [self.tokenizer.eos_token_id]

        return {"input_ids": input_ids, "output_ids": output_ids}

# Create the dataset and dataloader
dataset = ControlsDataset(tokenizer, df)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Set up the training loop
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epochs = 3  # Number of epochs to train for
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = torch.tensor(batch['input_ids']).to(device)
        output_ids = torch.tensor(batch['output_ids']).to(device)

        outputs = model(input_ids, labels=output_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
