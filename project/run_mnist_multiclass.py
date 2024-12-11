from mnist import MNIST

import minitorch

mndata = MNIST("project/data/")
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    # Create a parameter initialized randomly around zero.
    init = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(init)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # Flatten and multiply by weights, then add bias.
        bsz, features = x.shape
        w_reshaped = self.weights.value.view(features, self.out_size)
        result = x.view(bsz, features) @ w_reshaped
        shifted = result.view(bsz, self.out_size) + self.bias.value
        return shifted


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        # Convolution filters and biases
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        # Apply a 2D convolution followed by adding a bias term
        conv_result = minitorch.conv2d(input, self.weights.value)
        shifted = conv_result + self.bias.value
        return shifted


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    Steps:
    1. Convolve input with a 3x3 kernel, 4 output channels. ReLU the result (saved to self.mid).
    2. Convolve result with another 3x3 kernel, 8 output channels. ReLU the result (saved to self.out).
    3. Apply a 4x4 2D pooling (max or average, here max).
    4. Flatten into shape BATCH x 392.
    5. Linear -> 64 units, ReLU, then Dropout (25%).
    6. Linear -> C units.
    7. logsoftmax along the class dimension.
    """

    def __init__(self):
        super().__init__()
        self.mid = None
        self.out = None

        # First convolution: input is single-channel, output is 4-channel
        self.conv1 = Conv2d(1, 4, 3, 3)
        # Second convolution: takes in 4-channel from above, outputs 8-channel
        self.conv2 = Conv2d(4, 8, 3, 3)

        # Fully connected layers
        self.linear1 = Linear(392, 64)
        self.linear2 = Linear(64, C)

    def forward(self, x):
        # First convolution layer + ReLU
        after_conv1 = self.conv1(x)
        self.mid = after_conv1.relu()

        # Second convolution layer + ReLU
        after_conv2 = self.conv2(self.mid)
        self.out = after_conv2.relu()

        # 4x4 max pooling
        pooled = minitorch.maxpool2d(self.out, (4, 4))

        # Flatten to BATCH x 392
        flattened = pooled.view(pooled.shape[0], 392)

        # Linear layer to 64, ReLU, and Dropout
        h1 = self.linear1(flattened).relu()
        dropped = minitorch.dropout(h1, 0.25, ignore=not self.training)

        # Second linear layer to C
        logits = self.linear2(dropped)

        # Log-softmax over classes
        return minitorch.logsoftmax(logits, dim=1)


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        # One-hot encoding
        target = [0.0] * 10
        target[y] = 1.0
        ys.append(target)
        img = [[images[i][h * W + w] for w in range(W)] for h in range(H)]
        X.append(img)
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        # Run a single example through the model
        inp = minitorch.tensor([x], backend=BACKEND)
        return self.model.forward(inp)

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()  # re-initialize model
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(model.parameters(), learning_rate)
        losses = []

        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            model.train()

            # Iterate over training batches
            for batch_num, start_idx in enumerate(range(0, n_training_samples, BATCH)):
                if n_training_samples - start_idx <= BATCH:
                    # Skip incomplete batch
                    continue

                # Get batch data
                y_batch = y_train[start_idx : start_idx + BATCH]
                x_batch = X_train[start_idx : start_idx + BATCH]

                y_tensor = minitorch.tensor(y_batch, backend=BACKEND)
                x_tensor = minitorch.tensor(x_batch, backend=BACKEND)

                x_tensor.requires_grad_(True)
                y_tensor.requires_grad_(True)

                # Forward pass through model
                out = model.forward(x_tensor.view(BATCH, 1, H, W)).view(BATCH, C)

                # Compute loss as negative log-likelihood for the correct classes
                # prob is sum over classes where y=1
                prob = (out * y_tensor).sum(1)
                loss = -(prob / y_tensor.shape[0]).sum()
                loss.view(1).backward()

                assert loss.backend == BACKEND
                total_loss += loss[0]
                losses.append(total_loss)

                # Update parameters
                optim.step()

                # Every few batches, evaluate on validation subset
                if batch_num % 5 == 0:
                    model.eval()
                    correct = 0
                    # Evaluate on a single validation batch for quick feedback
                    val_example_num = 0
                    y_val_batch = y_val[val_example_num : val_example_num + BATCH]
                    x_val_batch = X_val[val_example_num : val_example_num + BATCH]
                    yv_tensor = minitorch.tensor(y_val_batch, backend=BACKEND)
                    xv_tensor = minitorch.tensor(x_val_batch, backend=BACKEND)

                    val_out = model.forward(xv_tensor.view(BATCH, 1, H, W)).view(BATCH, C)

                    # Accuracy calculation
                    for i in range(BATCH):
                        max_val = -1e9
                        pred = -1
                        for j in range(C):
                            if val_out[i, j] > max_val:
                                max_val = val_out[i, j]
                                pred = j
                        if yv_tensor[i, pred] == 1.0:
                            correct += 1

                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
