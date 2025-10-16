import os
import numpy as np
import matplotlib.pyplot as plt

this_file_path = os.path.dirname(os.path.realpath(__file__))
params_path = os.path.join(this_file_path, '../../02_training/02_training')

bootstraps = 10
initial_weights = 3
epochs = 30

plt.clf()
f, axes = plt.subplots(4, 2, figsize=(10, 16))

for training_config in [
  (0, 0, 'A1', 'binADNI_weights_30_epochs__new_normalizer/T1_@MNI_nlin',),
  (0, 1, 'A2', 'binADNI_weights_30_epochs__new_normalizer/T1_@MNI_nlin_bet',),
  (1, 0, 'B1', 'binADNI_weights_30_epochs__new_normalizer/T1_@MNI_nlin_bin01375',),
  (1, 1, 'B2', 'binADNI_weights_30_epochs__new_normalizer/T1_@MNI_nlin_bet_bin01375',),
  (2, 0, 'C1', 'binADNI_weights_30_epochs__new_normalizer/T1_@MNI_nlin_bin0275',),
  (2, 1, 'C2', 'binADNI_weights_30_epochs__new_normalizer/T1_@MNI_nlin_bet_bin0275',),
  (3, 0, 'D1', 'binADNI_weights_30_epochs__new_normalizer/T1_@MNI_nlin_bin04125',),
  (3, 1, 'D2', 'binADNI_weights_30_epochs__new_normalizer/T1_@MNI_nlin_bet_bin04125',),
]:
  (row_index, column_index, model_name, params_destination,) = training_config
  is_RG = params_destination.endswith('_RG') or params_destination.endswith('_RG_MNI') or params_destination.endswith('_RG_BG')

  params_destination_full_path = os.path.join(params_path, params_destination)
  if not os.path.exists(params_destination_full_path):
    continue

  param_names = os.listdir(params_destination_full_path)
  param_names.sort()

  data = np.zeros((30, 4, 30))

  # T1__boostrap_index-01__initial_weights_index-03__055-tcl-0.336-vcl-0.505-tca-0.845-vca-0.778-tsrim-0.991-vsrim-0.992.h5
  for param_name in param_names:
    bi = int(param_name[len('T1__bootstrap_index-'):len('T1__bootstrap_index-') + 2]) - 1
    iwi = int(param_name[len('T1__bootstrap_index-01__initial_weights_index-'):len('T1__bootstrap_index-01__initial_weights_index-') + 2]) - 1
    index_runner = iwi * 10 + bi
 
    epoch = int(param_name[len('T1__bootstrap_index-01__initial_weights_index-01__'):len('T1__bootstrap_index-01__initial_weights_index-01__') + 3])
    data[epoch - 1, 0, index_runner] = float(param_name[param_name.index('tcl-') + len('tcl-'):param_name.index('vcl-') - 1])
    data[epoch - 1, 1, index_runner] = float(param_name[param_name.index('vcl-') + len('vcl-'):param_name.index('tca-') - 1])
    data[epoch - 1, 2, index_runner] = float(param_name[param_name.index('tca-') + len('tca-'):param_name.index('vca-') - 1])
    data[epoch - 1, 3, index_runner] = float(param_name[param_name.index('vca-') + len('vca-'):param_name.index('tim--') - 1 if is_RG else param_name.index('.pth')])

  r = range (1, 31)
  m = data.mean(axis=2)
  std = data.std(axis=2)

  p_to = axes[row_index, column_index]

  # plt.clf()
  p_to.fill_between(r, m[:, 2] - std[:, 2], m[:, 2] + std[:, 2], facecolor='blue', alpha=0.5)
  p_to.plot(r, m[:, 2], color='blue', label='training')
  p_to.fill_between(r, m[:, 3] - std[:, 3], m[:, 3] + std[:, 3], facecolor='orange', alpha=0.5)
  p_to.plot(r, m[:, 3], color='orange', label='validation')
 
  p_to.set_ylim([0.45, 1.15])

  p_to.set_title(model_name)
  p_to.set_xlabel('Epoch')
  p_to.set_ylabel('Accuracy')

  p_to.legend(loc='lower right')


# plt.axis('scaled')

plt.tight_layout()
plt.savefig('accuracy_curves.png', dpi=250)
