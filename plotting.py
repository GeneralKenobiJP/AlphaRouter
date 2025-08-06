from matplotlib import pyplot as plt


def plot(x_axis, values, label, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, values, label=label, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def moving_average(values, window):
    ma_values = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        ma_values.append(sum(window_values) / len(window_values))

    return ma_values
