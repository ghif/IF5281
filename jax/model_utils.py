import jax
import jax.numpy as jnp
from flax import nnx

# LeNet5 (original LeCun et al., 1998)
class LeNet5(nnx.Module):
    def __init__(self, c, num_classes, rngs: nnx.Rngs):
        self.layer1 = nnx.Sequential(
            nnx.Conv(c, 6, kernel_size=(5, 5), rngs=rngs),
            nnx.tanh,
            lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        )
        
        self.layer2 = nnx.Sequential(
            nnx.Conv(6, 16, kernel_size=(5, 5), rngs=rngs),
            nnx.tanh,
            lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        )
        
        self.fc1 = nnx.Linear(256, 120, rngs=rngs)
        self.fc2 = nnx.Linear(120, 84, rngs=rngs)
        self.fc3 = nnx.Linear(84, num_classes, rngs=rngs)
        
    def __call__(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = h.reshape(h.shape[0], -1) # flatten
        h = nnx.relu(self.fc1(h))
        h = nnx.relu(self.fc2(h))
        logits = self.fc3(h)
        return logits

class ConvBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, replicate=False, rngs=None):
        self.layer1 = nnx.Sequential(
            nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), padding=1, rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
            nnx.relu
        )

        self.replicate = replicate
        if replicate:
            self.layer_r = nnx.Sequential(
                nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), padding=1, rngs=rngs),
                nnx.BatchNorm(out_channels, rngs=rngs),
                nnx.relu
            )
        self.layer2 = nnx.Sequential(
            nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), padding=1, rngs=rngs),
            nnx.BatchNorm(out_channels, rngs=rngs),
            nnx.relu, 
            lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        )
        
    def __call__(self, x):
        out = self.layer1(x)
        if self.replicate:
            out = self.layer_r(out)
        out = self.layer2(out)
        return out
        
# Define model
class VGG16(nnx.Module):
    def __init__(self, c, num_classes=10, rngs: nnx.Rngs = None):
        nchannels1 = [c, 64, 128]
        layers = []
        for i in range(len(nchannels1) - 1):
            in_channels = nchannels1[i]
            out_channels = nchannels1[i + 1]
            layers.append(ConvBlock(in_channels, out_channels, rngs=rngs))

        nchannels2 = [128, 256, 512, 512]
        for i in range(len(nchannels2) - 1):
            in_channels = nchannels2[i]
            out_channels = nchannels2[i + 1]
            layers.append(ConvBlock(in_channels, out_channels, replicate=True, rngs=rngs))
        
        self.conv_blocks = layers

        self.fc1 = nnx.Linear(512, 4096, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.5, rngs=rngs)
        self.fc2 = nnx.Linear(4096, 4096, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.5, rngs=rngs)
        self.fc3 = nnx.Linear(4096, num_classes, rngs=rngs)
        
    def __call__(self, x):
        out = x
        for block in self.conv_blocks:
            out = block(out)
        
        out = out.reshape(out.shape[0], -1)
        out = nnx.relu(self.fc1(out))
        out = self.dropout1(out)
        out = nnx.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

class ResidualBlock(nnx.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        rngs: nnx.Rngs = None
    ):
        self.downsample_layer = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_layer = nnx.Sequential(
                nnx.Conv(
                    in_channels, 
                    out_channels,
                    kernel_size=(1, 1),
                    strides=stride,
                    use_bias=False,
                    rngs=rngs
                ),
                nnx.BatchNorm(out_channels, rngs=rngs),
            )
        
        self.conv1 = nnx.Conv(
            in_channels, 
            out_channels, 
            kernel_size=(3, 3), 
            strides=stride, 
            padding=1,
            use_bias=False,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        
        self.conv2 = nnx.Conv(
            out_channels, 
            out_channels,
            kernel_size=(3, 3), 
            padding=1,
            use_bias=False,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample_layer is not None:
            identity = self.downsample_layer(x)
        
        out += identity
        out = nnx.relu(out)
        return  out

def count_parameters(model: nnx.Module):
    # In NNX, we can get state and count leaves of specific types
    state = nnx.state(model)
    return sum(x.size for x in jax.tree_util.tree_leaves(state) if hasattr(x, 'size'))


def make_residual_layer(
        in_channels: int, 
        out_channels: int, 
        stride: int, 
        num_layer: int,
        rngs: nnx.Rngs = None
    ):
    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride=stride, rngs=rngs))
    for _ in range(1, num_layer):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1, rngs=rngs))
    return layers


class ResNet(nnx.Module):
    def __init__(self, 
            img_channel: int, 
            in_channels: int,
            num_classes: int, 
            nlayers: list[int],
            rngs: nnx.Rngs = None
        ):
        in_channels = 64
        self.conv1 = nnx.Conv(
            img_channel,
            in_channels,
            kernel_size=(7, 7),
            strides=2,
            padding=3,
            use_bias=False,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(in_channels, rngs=rngs)
        self.maxpool = lambda x: nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        self.res_layer1 = make_residual_layer(in_channels, 64, 1, nlayers[0], rngs=rngs)
        self.res_layer2 = make_residual_layer(64, 128, 2, nlayers[1], rngs=rngs)
        self.res_layer3 = make_residual_layer(128, 256, 2, nlayers[2], rngs=rngs)
        self.res_layer4 = make_residual_layer(256, 512, 2, nlayers[3], rngs=rngs)

        self.fc = nnx.Linear(512, num_classes, rngs=rngs)

    def __call__(self, x):
        h = nnx.relu(self.bn1(self.conv1(x)))
        h = self.maxpool(h)
        
        for layer in self.res_layer1: h = layer(h)
        for layer in self.res_layer2: h = layer(h)
        for layer in self.res_layer3: h = layer(h)
        for layer in self.res_layer4: h = layer(h)

        h = jnp.mean(h, axis=(1, 2)) # Global Average Pooling
        logits = self.fc(h)
        return logits
    
# Define model
class SimpleRNN(nnx.Module):
    def __init__(self, input_size, hidden_size, output_size, rngs: nnx.Rngs = None):
        # Flax NNX doesn't have a built-in "RNN" equivalent yet like PyTorch's nn.RNN
        # We might need to implement it manually or use flax.linen.RNN
        # However, for simplicity, let's assume we want to mock the behavior
        self.rnn_cell = nnx.RNNCell(input_size, hidden_size, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, output_size, rngs=rngs)
        self.hidden_size = hidden_size
    
    def __call__(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        h = jnp.zeros((batch_size, self.hidden_size))
        
        h_list = []
        for t in range(seq_len):
            h = nnx.relu(self.rnn_cell(h, x[:, t, :]))
            h_list.append(h)
        
        # Stack hidden states
        h_stacked = jnp.stack(h_list, axis=1)
        y = self.linear(h_stacked)
        return y
    
class SimpleLSTM(nnx.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, rngs: nnx.Rngs = None):
        # For simplicity, assuming 1 layer for now
        self.lstm_cell = nnx.LSTMCell(input_size, hidden_size, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, output_size, rngs=rngs)
        self.hidden_size = hidden_size
    
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        h = jnp.zeros((batch_size, self.hidden_size))
        c = jnp.zeros((batch_size, self.hidden_size))
        
        h_list = []
        for t in range(seq_len):
            h, c = self.lstm_cell( (h, c), x[:, t, :])
            h_list.append(h)
        
        h_stacked = jnp.stack(h_list, axis=1)
        y = self.linear(h_stacked)
        return y
    
    def get_states_across_time(self, x):
        batch_size, seq_len, _ = x.shape
        h = jnp.zeros((batch_size, self.hidden_size))
        c = jnp.zeros((batch_size, self.hidden_size))
        
        h_list, c_list = [], []
        for t in range(seq_len):
            h, c = self.lstm_cell((h, c), x[:, t, :])
            h_list.append(h)
            c_list.append(c)
        
        return jnp.concatenate(h_list), jnp.concatenate(c_list)

class SimpleBigram(nnx.Module):
    def __init__(self, vocab_size, seq_len, n_embed, n_hidden, num_layers=1, rngs: nnx.Rngs = None):
        self.seq_len = seq_len
        self.embedding = nnx.Embed(vocab_size, n_embed, rngs=rngs)
        self.lstm = nnx.LSTMCell(n_embed, n_hidden, rngs=rngs)
        self.linear = nnx.Linear(n_hidden, vocab_size, rngs=rngs)
        self.n_hidden = n_hidden
    
    def __call__(self, idx):
        # idx shape: (batch, seq_len)
        x = self.embedding(idx) # (batch, seq_len, n_embed)
        
        batch_size, seq_len, _ = x.shape
        h = jnp.zeros((batch_size, self.n_hidden))
        c = jnp.zeros((batch_size, self.n_hidden))
        
        h_list = []
        for t in range(seq_len):
            h, c = self.lstm((h, c), x[:, t, :])
            h_list.append(h)
        
        h_stacked = jnp.stack(h_list, axis=1) # (batch, seq_len, n_hidden)
        logits = self.linear(h_stacked)
        return logits
    
    def generate(self, idx, max_new_tokens, rngs: nnx.Rngs):
        # Initial call to get the state would be complex in the loop if we don't return state
        # But for simplicity, let's just do it like the PyTorch version (inefficient but works)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] # last time step
            
            # Sample
            probs = jax.nn.softmax(logits, axis=-1)
            # Use jax.random.choice for sampling
            rngs.next()
            idx_next = jax.random.choice(rngs.params(), jnp.arange(probs.shape[-1]), p=probs[0])
            idx_next = idx_next.reshape(1, 1)
            idx = jnp.concatenate([idx, idx_next], axis=1)
            
        return idx
