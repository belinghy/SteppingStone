import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
plt.show(block=False)
ax1 = fig.add_subplot(111)

with open("CassieStepper-v1_sampling_prob85.pkl", "rb") as fp:
	sampling_prob_list = pickle.load(fp)

sampling_prob_list = np.stack(sampling_prob_list)
print(sampling_prob_list.shape)

sampling_sum = np.sum(sampling_prob_list[:, :, :], axis=1)
print(sampling_sum.shape)

for i in range(sampling_sum.shape[1]):
	ax1.plot(sampling_sum[:, i], label="{}".format(i-5))
ax1.legend()
plt.show()