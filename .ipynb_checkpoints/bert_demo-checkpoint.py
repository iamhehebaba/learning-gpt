
	
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BertConfig:
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, max_position_embeddings=512,
                 type_vocab_size=16, initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads!= 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                    config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # attention_mask = attention_mask.unsqueeze(1).expand_as(attention_scores)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == 'gelu':
            self.intermediate_act_fn = self.gelu
        else:
            raise ValueError("Unsupported activation function: %s" % config.hidden_act)

    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, attention_mask)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return pooled_output, sequence_output, encoded_layers
if __name__ == "__main__":
    
    bert_config = BertConfig(100)
    bert_model = BertModel(bert_config)
    print(bert_model)
	
	
	
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "segment_ids": torch.tensor(item["segment_ids"]),
            "masked_lm_ids": torch.tensor(item["masked_lm_ids"]),
            "masked_lm_positions": torch.tensor(item["masked_lm_positions"]),
            "masked_lm_weights": torch.tensor(item["masked_lm_weights"]),
            "next_sentence_labels": torch.tensor(item["next_sentence_labels"])
        }


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, label_ids, label_weights):
    input_tensor = gather_indexes(input_tensor, positions)
    sequential = nn.Sequential(
        nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        nn.LayerNorm(bert_config.hidden_size),
        nn.ReLU()
    )
    input_tensor = sequential(input_tensor)  # 使用sequential处理input_tensor
    output_bias = nn.Parameter(torch.zeros(bert_config.vocab_size))
    logits = torch.matmul(input_tensor, output_weights.transpose(0, 1)) + output_bias
    log_probs = nn.functional.log_softmax(logits, dim=-1)

    label_ids = label_ids.reshape(-1)
    label_weights = label_weights.reshape(-1)
    one_hot_labels = torch.nn.functional.one_hot(label_ids, num_classes=bert_config.vocab_size).float()
    per_example_loss = -torch.sum(log_probs * one_hot_labels, dim=-1)
    numerator = torch.sum(label_weights * per_example_loss)
    denominator = torch.sum(label_weights) + 1e-5
    loss = numerator / denominator

    return loss, per_example_loss, log_probs


def get_next_sentence_output(bert_config, input_tensor, labels):
    output_weights = nn.Parameter(torch.randn(2, bert_config.hidden_size))
    output_bias = nn.Parameter(torch.zeros(2))
    logits = torch.matmul(input_tensor, output_weights.transpose(0, 1)) + output_bias
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    labels = labels.reshape(-1)
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
    per_example_loss = -torch.sum(one_hot_labels * log_probs, dim=-1)
    loss = torch.mean(per_example_loss)
    return loss, per_example_loss, log_probs


def gather_indexes(sequence_tensor, positions):
    sequence_shape = list(sequence_tensor.shape)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = torch.arange(0, batch_size, dtype=torch.int64).reshape(-1, 1) * seq_length
    flat_positions = (positions + flat_offsets).reshape(-1)
    flat_sequence_tensor = sequence_tensor.reshape(batch_size * seq_length, width)
    output_tensor = flat_sequence_tensor[flat_positions]
    return output_tensor


def train(model, train_loader, bert_config, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        masked_lm_positions = batch["masked_lm_positions"]
        masked_lm_ids = batch["masked_lm_ids"]
        masked_lm_weights = batch["masked_lm_weights"]
        next_sentence_labels = batch["next_sentence_labels"]

        optimizer.zero_grad()
        _, sequence_output, _ = model(input_ids, segment_ids)

        masked_lm_loss, _, _ = get_masked_lm_output(bert_config, sequence_output, model.embeddings.word_embeddings.weight,
                                                    masked_lm_positions, masked_lm_ids, masked_lm_weights)
        next_sentence_loss, _, _ = get_next_sentence_output(bert_config, model.pooler(sequence_output),
                                                             next_sentence_labels)

        loss = masked_lm_loss + next_sentence_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, bert_config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            masked_lm_positions = batch["masked_lm_positions"]
            masked_lm_ids = batch["masked_lm_ids"]
            masked_lm_weights = batch["masked_lm_weights"]
            next_sentence_labels = batch["next_sentence_labels"]

            _, sequence_output, _ = model(input_ids, segment_ids)

            masked_lm_loss, _, _ = get_masked_lm_output(bert_config, sequence_output, model.embeddings.word_embeddings.weight,
                                                        masked_lm_positions, masked_lm_ids, masked_lm_weights)
            next_sentence_loss, _, _ = get_next_sentence_output(bert_config, model.pooler(sequence_output),
                                                                 next_sentence_labels)

            loss = masked_lm_loss + next_sentence_loss
            total_loss += loss.item()
    return total_loss / len(val_loader)


# def main():
bert_config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=16,
    initializer_range=0.02
)
model = BertModel(bert_config)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

train_dataset = MyDataset('output_data.json')
val_dataset = MyDataset('output_data.json')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

num_train_steps = 100000
num_warmup_steps = 10000

# for epoch in range(num_train_steps):
#     train_loss = train(model, train_loader, bert_config, optimizer)
#     val_loss = evaluate(model, val_loader, bert_config)
#     print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

# if __name__ == '__main__':
#     main()