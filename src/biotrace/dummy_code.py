import matplotlib.pyplot as plt
import numpy as np

def plot_function1(x):
    y = np.sin(x)
    plt.plot(x, y, label='Plot 1')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

def plot_function2(x):
    y = np.cos(x)
    plt.plot(x, y, label='Plot 2', color='red')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

if __name__ == "__main__":
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Generate x values for both plots
    x = np.linspace(0, 2 * np.pi, 100)

    # Call the first plotting function with the first subplot
    plt.sca(ax1)
    plot_function1(x)

    # Call the second plotting function with the second subplot
    plt.sca(ax2)
    plot_function2(x)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the combined plot
    plt.show()