#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here
people = ['Farrah', 'Fred', 'Felicia']
apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]

width = 0.5
x = np.arange(len(people))

plt.bar(x, apples, width, label='apples', color='red')
plt.bar(x, bananas, width, bottom=apples, label='bananas', color='yellow')
plt.bar(x, oranges, width, bottom=apples+bananas, label='oranges', color='#ff8000')
plt.bar(x, peaches, width, bottom=apples+bananas+oranges, label='peaches', color='#ffe5b4')

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(x, people)
plt.yticks(np.arange(0, 81, 10))
plt.legend()
plt.show()