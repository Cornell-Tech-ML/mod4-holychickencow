import random

import embeddings

import minitorch
from datasets import load_dataset

BACKEND = minitorch.TensorBackend(minitorch.FastOps)


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        bsz, in_features = x.shape
        flat_w = self.weights.value.view(in_features, self.out_size)
        # Linear transformation: xW + b
        lin_output = (x.view(bsz, in_features) @ flat_w).view(bsz, self.out_size)
        return lin_output + self.bias.value


class Conv1d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        # Initialize parameters for convolution weights and bias terms
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        # Perform a 1D convolution over the input using the stored weights
        # and then add the bias. The input should have shape: [batch, channels, length]
        conv_out = minitorch.conv1d(input, self.weights.value)
        # Add bias (broadcasting will handle dimension alignment)
        return conv_out + self.bias.value


class CNNSentimentKim(minitorch.Module):
    """
    A CNN-based sentiment analysis model following the approach described by Y. Kim (2014).

    Steps:
    1. Take embedded inputs of shape [batch, sentence_length, embedding_dim],
       and permute to [batch, embedding_dim, sentence_length].
    2. Pass through multiple Conv1d filters of different kernel sizes (3,4,5), each producing
       feature maps of size `feature_map_size`, then apply ReLU.
    3. Perform max-over-time pooling across each feature map dimension.
    4. Combine the pooled results (here we sum them, but concatenation followed by linear
       could also work if desired).
    5. Apply a linear layer to the pooled representation.
    6. Apply dropout (if training) and then a sigmoid to produce final class probabilities.
    """

    def __init__(
        self,
        feature_map_size=100,
        embedding_size=50,
        filter_sizes=[3, 4, 5],
        dropout=0.25,
    ):
        super().__init__()
        self.dropout = dropout

        # Three convolutional filters with different kernel widths
        self.conv1 = Conv1d(in_channels=embedding_size, out_channels=feature_map_size, kernel_width=filter_sizes[0])
        self.conv2 = Conv1d(in_channels=embedding_size, out_channels=feature_map_size, kernel_width=filter_sizes[1])
        self.conv3 = Conv1d(in_channels=embedding_size, out_channels=feature_map_size, kernel_width=filter_sizes[2])

        # A fully connected layer that maps from the pooled features to a single output
        self.fc = Linear(feature_map_size, 1)

    def forward(self, embeddings):
        # Input embeddings: [batch, sentence_length, embedding_dim]
        # Permute for Conv1d: [batch, embedding_dim, sentence_length]
        rearranged = embeddings.permute(0, 2, 1)

        # Apply each convolution + ReLU separately, then max-pool over the time dimension
        act1 = self.conv1(rearranged).relu()
        act2 = self.conv2(rearranged).relu()
        act3 = self.conv3(rearranged).relu()

        pool1 = minitorch.max(act1, dim=2)
        pool2 = minitorch.max(act2, dim=2)
        pool3 = minitorch.max(act3, dim=2)

        # Merge (in this case we sum them, which preserves shape)
        merged = pool1 + pool2 + pool3

        # Fully connected layer, ReLU, and Dropout
        fc_out = self.fc(merged.view(merged.shape[0], merged.shape[1])).relu()
        dropped = minitorch.dropout(fc_out, p=self.dropout, ignore=not self.training)

        # Sigmoid activation for output
        final_out = dropped.sigmoid().view(embeddings.shape[0])
        return final_out


# Evaluation helper methods
def get_predictions_array(y_true, model_output):
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        predicted_label = 1.0 if logit > 0.5 else 0.0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array):
    correct = 0
    for y_true, y_pred, _logit in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)


best_val = 0.0


def default_log_fn(
    epoch,
    train_loss,
    losses,
    train_predictions,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    global best_val
    if validation_accuracy and validation_accuracy[-1] > best_val:
        best_val = validation_accuracy[-1]
    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")


class SentenceSentimentTrain:
    def __init__(self, model):
        self.model = model

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=10,
        max_epochs=100000,  # Adjusted to match the provided final code
        data_val=None,
        log_fn=default_log_fn,
    ):
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []

        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(range(0, n_training_samples, batch_size)):
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)

                out = model.forward(x)
                # Compute binary cross-entropy-like loss
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()

                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]

                optim.step()

            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(y_val, backend=BACKEND)
                x = minitorch.tensor(X_val, backend=BACKEND)
                val_out = model.forward(x)
                validation_predictions += get_predictions_array(y, val_out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)

            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )

            total_loss = 0.0


def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0 vectors up to max_sentence_len
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    # Find the maximum sentence length to pad/truncate to
    max_sentence_len = 0
    for s in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(s.split()))

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for _ in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    train_size = 450
    validation_size = 100
    learning_rate = 0.01
    max_epochs = 100000

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )
