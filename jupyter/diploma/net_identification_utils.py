from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
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
class ControlDataset(Dataset):

    def __init__(self, global_u, global_y):
        assert len(global_u) == len(global_y)
        self.global_u = global_u
        self.global_y = global_y

    def __getitem__(self, index):
        return self.global_u[index], self.global_y[index]

    def __len__(self):
        return len(self.global_u)


class ControlDatasetWithOutputTrain(Dataset):
    def __init__(self, u, output, window_size, layer_input_size):
        self.input = []
        self.output_to_train = []
        self.output_to_predict = []
        for input_sample, output_sample in zip(u, output):
            appendix_size = window_size - input_sample.size()[0]
            self.input.append(torch.cat((torch.zeros([appendix_size], requires_grad=True), input_sample), 0))
            self.output_to_train.append(
                torch.cat((torch.zeros([appendix_size], requires_grad=True), output_sample[:-1 * layer_input_size]), 0)
            )
            self.output_to_predict.append(output_sample[-1 * layer_input_size:])

    def __getitem__(self, index):
        return self.input[index], self.output_to_train[index], self.output_to_predict[index]

    def __len__(self):
        return len(self.input)


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


def generate_data_for_input_output_rnn_training(
        control_sys,
        impact_time,
        n_signals,
        n_samples_per_signal,
        sample_u_size,
        layer_input_size,
        u_signal_generation_func
):
    assert n_samples_per_signal % 10 == 0
    assert sample_u_size % layer_input_size == 0
    samples_u = list()
    samples_response = list()
    for _ in range(n_signals):
        u = u_signal_generation_func(impact_time)
        response = control.forced_response(control_sys, T=np.arange(impact_time), U=u)[1][0]

        # max length signals
        for _ in range(n_samples_per_signal // 2):
            start = np.random.randint(
                low=0,
                high=impact_time - sample_u_size
            )
            fin = start + sample_u_size
            samples_u.append(torch.tensor(u[start: fin], requires_grad=True).float())
            samples_response.append(torch.tensor(response[start: fin], requires_grad=True).float())

        # lower length signals
        for reccurency_depth in range(1, sample_u_size // layer_input_size + 1):
            for _ in range(n_samples_per_signal // 10):
                start = np.random.randint(
                    low=0,
                    high=impact_time - sample_u_size
                )
                fin = start + reccurency_depth * layer_input_size
                samples_u.append(torch.tensor(u[start: fin], requires_grad=True).float())
                samples_response.append(torch.tensor(response[start: fin], requires_grad=True).float())

    return ControlDatasetWithOutputTrain(samples_u, samples_response, sample_u_size, layer_input_size)


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

    def forward(self, system_input_signal):
        hidden, _ = self.layers[0](system_input_signal.view(-1, self.reccurency, self.layer_input_size).transpose(0, 1))
        last_hidden = hidden[-1, :, :]
        return self.layers[1](last_hidden)


# ancillary utils for training
def train(model, epochs, train_loader, loss):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        losses = []
        for data_tuple in train_loader:
            prediction = model(*data_tuple[:-1])
            loss_batch = loss(prediction, data_tuple[-1])
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


class ControlLSTMInputsOutputs(nn.Module):
    def __init__(self, window_size, layer_input_size, hidden_size, output_size, num_layers=2):
        assert window_size % layer_input_size == 0
        super().__init__()
        self.layer_input_size = layer_input_size
        self.reccurency_depth = window_size // layer_input_size
        self.hidden_size = hidden_size

        self.input_processing_cell = torch.nn.LSTM(
            input_size=layer_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.output_processing_cell = torch.nn.LSTM(
            input_size=layer_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.output_layer = (torch.nn.Linear(hidden_size * 2, output_size))

    def forward(self, system_input_signal, system_output_signal=None):
        assert system_output_signal is not None or \
               system_input_signal.size()[-1] == self.layer_input_size

        input_processing_hidden, _ = self.input_processing_cell(
            system_input_signal.view(
                # <batch_size>, <reccurency depth>, <one lstm cell input size>
                -1, system_input_signal.size()[-1] // self.layer_input_size, self.layer_input_size
                # transposing because view is incorrect if passing required shape to view directly
            ).transpose(0, 1)
        )
        if system_output_signal is not None:
            output_processing_hidden, _ = self.output_processing_cell(
                system_output_signal.view(
                    # <batch_size>, <reccurency depth>, <one lstm cell input size>
                    -1, system_output_signal.size()[-1] // self.layer_input_size, self.layer_input_size
                    # transposing because view is incorrect if passing required shape to view directly
                ).transpose(0, 1)
            )
        else:
            output_processing_hidden = torch.zeros(input_processing_hidden.size())

        last_hidden = torch.cat((input_processing_hidden[-1, :, :], output_processing_hidden[-1, :, :]), -1)

        return self.output_layer(last_hidden)

    def predict(self, system_input_signal):
        if system_input_signal.ndimension() == 1:
            system_input_signal = system_input_signal.view(1, -1)
        assert system_input_signal.size()[1] % self.layer_input_size == 0
        n_atomic_parts = system_input_signal.size()[1] // self.layer_input_size
        predicted_outputs = torch.zeros(*system_input_signal.size())
        for i in range(n_atomic_parts):
            if i == 0:
                predicted_outputs[:, i * self.layer_input_size: (i + 1) * self.layer_input_size] = self.forward(
                    system_input_signal[:, :20]
                )
            else:
                prediction_data_start = max(0, i - self.reccurency_depth + 1) * self.layer_input_size
                predicted_outputs[:, i * self.layer_input_size: (i + 1) * self.layer_input_size] = self.forward(
                    system_input_signal[:, prediction_data_start: (i + 1) * self.layer_input_size],
                    predicted_outputs[:, prediction_data_start:i * self.layer_input_size]
                )
        return predicted_outputs

    def test(self, test_dataset, loss, batch_size=10):
        relevant_testing_data_start_position = self.layer_input_size * (self.reccurency_depth - 1)
        assert test_dataset[0][0].size()[0] > relevant_testing_data_start_position
        losses = []
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        for input_signal, output_signal in test_dataloader:
            prediction = self.predict(input_signal)
            loss_batch = loss(
                prediction[:, relevant_testing_data_start_position:],
                output_signal[:, relevant_testing_data_start_position:]
            )
            losses.append(loss_batch.item())
        return np.mean(losses)
