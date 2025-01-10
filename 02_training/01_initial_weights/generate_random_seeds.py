import random
import json

random_seeds = []

for _ in range(6):
  random_seeds.append(random.randint(0, 10000))

with open('random_seeds.json', 'w') as outfile:
  json.dump(random_seeds, outfile)
