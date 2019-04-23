from torch import nn, optim
from torch.utils.data import dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import control


# discrete output visualisation
def _color_gen():
    colors = ['green', 'blue', 'red']
    i = 0
    while True:
        yield colors[i]
        i += 1
        i = i % len(colors)


def plot_discrete_output(file_name=None, **kwargs):
    for elem in kwargs.keys():
        if type(kwargs[elem]) == torch.Tensor:
            kwargs[elem] = kwargs[elem].detach().numpy().ravel()
    data_dict = kwargs
    assert len(set((elem.shape for elem in data_dict.values()))) == 1
    fig, ax = plt.subplots()
    for label, data, color in zip(data_dict.keys(), data_dict.values(), _color_gen()):
        # ax.set_ylim((-0.5, 0.5))
        ax.set_xticks(np.arange(data.shape[0]))
        ax.set_xlabel('Момент времени', fontsize=15)
        ax.set_ylabel('Выход системы', fontsize=15)
        ax.scatter(x=np.arange(data.shape[0]), y=data, label=label, color=color)

    ax.legend(fontsize=15, loc="upper right")
    ax.grid()
    counter = 0
    for tic in ax.xaxis.get_major_ticks():
        if counter % 10 != 0:
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        counter += 1
    fig.set_size_inches((10, 10))
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)


# ancillary class for pytorch
class ControlDataset(dataset.Dataset):

    def __init__(self, global_u, global_y):
        assert len(global_u) == len(global_y)
        self.global_u = global_u
        self.global_y = global_y

    def __getitem__(self, index):
        return self.global_u[index], self.global_y[index]

    def __len__(self):
        return len(self.global_u)


# function generate output and input by system and other params
def generate_data(control_sys, impact_time, cooldown_time, n_signals, u_signal_generation_func):
    total_time = impact_time + cooldown_time
    global_u = [
        np.concatenate((u_signal_generation_func(impact_time), np.zeros(cooldown_time)))
        for _ in range(n_signals)
    ]
    global_y = [control.forced_response(control_sys, T=np.arange(total_time), U=u)[1] for u in global_u]
    global_u = [torch.tensor(elem.ravel(), requires_grad=True).float() for elem in global_u]
    global_y = [torch.tensor(elem.ravel(), requires_grad=True).float() for elem in global_y]

    return ControlDataset(global_u, global_y)


def generate_data_for_rnn(
        control_sys,
        impact_time,
        n_signals,
        n_samples_per_signal,
        sample_u_size,
        sample_response_size,
        u_signal_generation_func
):
    samples_u = list()
    samples_response = list()
    for _ in range(n_signals):
        u = u_signal_generation_func(impact_time)
        response = control.forced_response(control_sys, T=np.arange(impact_time), U=u)[1][0]
        for _ in range(n_samples_per_signal):
            start = np.random.randint(
                low=0,
                high=impact_time - sample_u_size
            )
            fin = start + sample_u_size
            samples_u.append(torch.tensor(u[start: fin], requires_grad=True).float())
            samples_response.append(torch.tensor(response[fin - sample_response_size: fin], requires_grad=True).float())

    return ControlDataset(samples_u, samples_response)


# simple full connected net
class FCnet(nn.Module):

    def __init__(self, input_size, output_size, n_layers, layer_size):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)])
        self.layers.extend(nn.Linear(layer_size, layer_size) for _ in range(n_layers - 2))
        self.layers.append(nn.Linear(layer_size, output_size))
        self.layer_size = layer_size
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=1)
        self.input_size = input_size
        self.layer_size = layer_size
        self.tanh = nn.Hardtanh(min_val=-1, max_val=1)

    def forward(self, x):
        y = self.tanh(self.layers[0](x))
        for i in range(1, len(self.layers)):
            y = self.tanh(self.layers[i](y))
        return y


# lstm net for system identification, use only system inputs
class ControlLSTMInputs(nn.Module):
    def __init__(self, window_size, layer_input_size, hidden_size, output_size, num_layers=2):
        assert window_size % layer_input_size == 0
        super().__init__()
        self.reccurency = window_size // layer_input_size
        self.layer_input_size = layer_input_size
        
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.LSTM(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
        )
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, window_data):
        hidden, _ = self.layers[0](window_data.view(-1, self.reccurency, self.layer_input_size).transpose(0, 1))
        last_hidden = hidden[-1, :, :]
        return self.layers[1](last_hidden)


# ancillary utils for training
def train(model, epochs, train_loader, loss):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        losses = []
        for x, y in train_loader:
            prediction = model(x)
            loss_batch = loss(prediction, y)
            losses.append(loss_batch.item())
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        print('epoch {e}, mse {m}'.format(e=epoch, m=np.mean(losses)))


def test(model, test_loader, loss):
    losses = []
    for x, y in test_loader:
        prediction = model(x)
        loss_batch = loss(prediction, y)
        losses.append(loss_batch.item())
    return np.mean(losses)
