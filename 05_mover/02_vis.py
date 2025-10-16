import json
import numpy as np
import matplotlib.pyplot as plt

stepper = range(1, 101, 3)

with open('values.json', 'r') as infile:
  groups = json.load(infile)

plt.clf()

plt.plot(stepper, np.array(groups['CN']).swapaxes(0, 1))
plt.plot(stepper,  np.array(groups['AD']).swapaxes(0, 1))

plt.savefig('divergence.png')

plt.clf()

group_CN = np.asarray(groups['CN'])
plt.plot(stepper, np.mean(group_CN, axis=0) * 100)

group_AD = np.asarray(groups['AD'])
plt.plot(stepper, np.mean(group_AD, axis=0) * 100)

# group difference
group_diff = np.mean(group_CN, axis=0) - np.mean(group_AD, axis=0)
group_diff /= np.max(group_diff)
plt.plot(stepper, group_diff * 100)

plt.plot([13.75, 13.75], [0, group_diff[4] * 100], 'o-', c='black')
plt.plot([27.50, 27.50], [0, group_diff[9] * 100], 'o--', c='black')
plt.plot([41.25, 41.25], [0, group_diff[14] * 100], 'o-.', c='black')
plt.plot([0, 100], [0, 0], '-', c='black')

plt.legend(['NC', 'AD', 'Rel. difference', 'B1/B2 (13.75%)', 'C1/C2 (27.5%)', 'D1/D2 (41.25%)'])

plt.xlim([0, 100])
plt.ylim([-15, 105])

plt.xlabel('Binarization thresholds in %')
plt.ylabel('Residual voxels in %')

plt.savefig('group_divergence.png', dpi=250, bbox_inches='tight')

print(np.argmax(np.mean(group_CN, axis=0) - np.mean(group_AD, axis=0)))

# print(stepper)



print(np.mean(group_CN, axis=0) - np.mean(group_AD, axis=0))

# print(np.mean(group_CN, axis=0))
# print(np.mean(group_AD, axis=0))
