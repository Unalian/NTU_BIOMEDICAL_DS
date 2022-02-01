import matplotlib.pyplot as plt
import numpy as np


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    for j in range(100):
        plt.subplot(10, 20, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img[i][j], cmap=plt.cm.binary)

    #   plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{}: {} {:2.0f}% ({})".format(i, predicted_label,
                                         100 * np.max(predictions_array),
                                         true_label),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(100))
    plt.yticks([])
    thisplot = plt.bar(range(100), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def setDataBag(y_set, x_set) :
    i, memSum = 0, 0
    x_toy, x_toy_bag, y_toy = [], [], []
    tag = 0
    for num in y_set:
        if num == 0 or num == 7:
            x_toy_bag.append(x_set[i])
            memSum += 1
            if num == 0:
                tag += 1
            if memSum % 100 == 0:
                x_toy.append(x_toy_bag)
                x_toy_bag = []
                y_toy.append(tag)
                tag = 0
        i += 1
    x_toy = np.array(x_toy)
    y_toy = np.array(y_toy)
    x_toy = x_toy / 255.0
    return x_toy, y_toy

