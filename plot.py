import matplotlib.pyplot as plt

data = open("result_file", "r").readlines()
scores = []
s_acc = []
for i, line in enumerate(data):
    if i % 2 == 0:
        scores.append(float(line))
    else:
        s_acc.append(float(line))


def plot(eval_episodes, y, title, x_name, y_name):
    x = [(x + 1) * eval_episodes for x in range(len(y))]
    plt.plot(x, y)
    plt.title(title)
#     plt.legend(legend)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
#     return plt
    plt.savefig("{}.jpg".format(title))
    plt.cla()

plot(2000, scores, "Scores Performance", "Episodes", "Score")
plot(2000, s_acc, "Success Accuracy Performance", "Episodes", "Success Accuracy")