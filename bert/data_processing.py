import json
import random
import collections
from transformers import BertTokenizer
class Args:
    def __init__(self):
        self.input_file = r"../data/bert_sample_text.txt"
        self.output_file = "../data/bert_output_data.json"
        self.vocab_file = r"/Users/wangaijun/pythoncode/github/model/bert-base-chinese/vocab.txt"
        self.do_lower_case = True
        self.do_whole_word_mask = False
        self.max_seq_length = 512
        self.max_predictions_per_seq = 20
        self.random_seed = 12345
        self.dupe_factor = 1
        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1

# 创建Args实例
args = Args()

class TrainingInstance:
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(self.tokens))
        s += "segment_ids: %s\n" % (" ".join(map(str, self.segment_ids)))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(map(str, self.masked_lm_positions)))
        s += "masked_lm_labels: %s\n" % (" ".join(self.masked_lm_labels))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()
		
def create_int_feature(values):
    return values

def create_float_feature(values):
    return values

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if args.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append((index, tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x[0])

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
#         print("p",p)
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()
			
def create_training_instances(input_files, tokenizer, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob, max_predictions_per_seq, rng):
    all_documents = []

    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as reader:
            while True:
                line = reader.readline()
                if not line:
                    break
                line = line.strip()

                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents.append(tokens)

    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instant=create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

            instances.extend(
              instant  )

    rng.shuffle(instances)
    return instances

def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    document = all_documents[document_index]

    max_num_tokens = max_seq_length - 3

    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
    
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)

                tokens.append("[SEP]")
                segment_ids.append(1)  
                    
                    
                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    
    return instances
	
def write_instance_to_json_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_file):
    all_features = []

    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)

        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_weights": masked_lm_weights,
            "next_sentence_labels": next_sentence_label
        }

        all_features.append(features)


    with open(output_file, "w", encoding="utf-8") as writer:
        for feature in all_features:
            writer.write(json.dumps(feature, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_features)} total instances to {output_file}")
	
def main():
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    input_files = args.input_file.split(",")
    print("*** Reading from input files ***")
    for input_file in input_files:
        print(f"  {input_file}")

    rng = random.Random(args.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, args.max_seq_length, args.dupe_factor,
        args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq, rng)

    output_file = args.output_file
    write_instance_to_json_files(instances, tokenizer, args.max_seq_length,
                                 args.max_predictions_per_seq, output_file)

if __name__ == "__main__":
    main()
	