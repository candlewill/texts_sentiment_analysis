import pandas as pd
import matplotlib.pyplot as plt

frequencies = [6, 16, 75, 160, 244, 260, 145, 73, 16, 4, 1]  # bring some raw data

freq_series = pd.Series.from_array(
    frequencies)  # in my original code I create a series and run on that, so for consistency I create a series from the list.

x_labels = [108300.0, 110540.0, 112780.0, 115020.0, 117260.0, 119500.0, 121740.0, 123980.0, 126220.0, 128460.0,
            130700.0]

# now to plot the figure...
plt.figure(figsize=(12, 8))
ax = freq_series.plot(kind='bar')
ax.set_title("Amount Frequency")
ax.set_xlabel("Amount ($)")
ax.set_ylabel("Frequency")
ax.set_xticklabels(x_labels)

rects = ax.patches

# Now make some labels
labels = ["label%d" % i for i in range(len(rects))]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

plt.savefig("./images/image.png")
