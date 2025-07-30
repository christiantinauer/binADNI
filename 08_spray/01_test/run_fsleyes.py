import subprocess
import json

with open('bin_bet.json') as f:
  clusters = json.load(f)

params = [
  'fsleyes',
]

for heatmap_path in clusters[1]:
  params.append(heatmap_path)

subprocess.call(params)
