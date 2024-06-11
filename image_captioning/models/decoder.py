import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Decoder(nn.Module):
    """
    Decoder:
        Use a vanilla one-to many LSTM architecture for generating the captions from the image features.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    # TO BE DELETED: train
    def forward(self, features, captions, lengths):
        """
        Forward propagation.
        @param
            features: Tensor of dimensions (batch_size, embed_size) representing the feature vectors of the input images
            captions: ?
            lengths: ?
            train: ?
        return:
            ???: ???
        """
        # TO BE REVIEWED: Embedding of the tokenized? captions
        embeddings = self.embed(captions)
        # Concatenate the features tensor obtained from the CNN with the embeddings of the captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # Use the lengths tensor to correctly handle padded tensors
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        return outputs

    def sample(self, features, states=None, beam_width=20):
        """
        Sampling function that generates the beam search approximation for the sequence S that maximizes the probability P(S|I), where I is the input image.
        @param
          features: Latent representation of the image I
          states: Previous hidden and cell states
          beam_width: Beam width used for the beam search
        """
        inputs = features.unsqueeze(1)

        # Beam search initialization. Notice that we manually handle the hidden and cell states
        sequences = [[[], 0.0, inputs, states]]

        for _ in range(self.max_seq_length):
            all_candidates = []
            # Iterate over the candidate sequences
            for seq, score, inputs, states in sequences:
                # Compute the log probabilities for the next word using as input the previous word in the sequence
                hiddens, states = self.lstm(inputs, states)
                outputs = self.linear(hiddens.squeeze(1))
                log_probs = torch.log_softmax(outputs, dim=1)

                # Get top beam_width candidate words
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=1)

                # Add the top beam_width candidate words to the candidate sequence and save the information in the all_candidates array
                for i in range(beam_width):
                    candidate = [seq + [top_indices[0][i].item()],
                                 (score + top_log_probs[0][i].item()) / len(seq + [top_indices[0][i].item()]),
                                 self.embed(top_indices[:, i]).unsqueeze(1),
                                 states]
                    all_candidates.append(candidate)

            # Order all candidates by score and select the top beam_width sequences
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]

        # Select the sequence with the highest score
        best_sequence = max(sequences, key=lambda x: x[1])[0]
        best_sequence = torch.tensor(best_sequence)
        return best_sequence.unsqueeze(0)