import os
import json

this_file_path = os.path.dirname(os.path.realpath(__file__))

bootstraps = 10
initial_weights = 3
epochs = 30

trainings = {}
trainings_highest = {}

for training_config in [
  # params destination            min training accuracy   min validation accuracy   fixed epoch   min relevance inner mask
  ('T1_@MNI_nlin',                .55,                    None,                      30,           None),
  ('T1_@MNI_nlin_bet',            .55,                    None,                      30,           None),
  ('T1_@MNI_nlin_bin01375',       .55,                    None,                      30,           None),
  ('T1_@MNI_nlin_bet_bin01375',   .55,                    None,                      30,           None),
  ('T1_@MNI_nlin_bin0275',        .55,                    None,                      30,           None),
  ('T1_@MNI_nlin_bet_bin0275' ,   .55,                    None,                      30,           None),
  ('T1_@MNI_nlin_bin04125',       .55,                    None,                      30,           None),
  ('T1_@MNI_nlin_bet_bin04125',   .55,                    None,                      30,           None),
]:
  (params_destination, min_training_accuracy, min_validation_accuracy, fixed_epoch, min_relevance_inner_mask) = training_config

  params_destination_full_path = os.path.join(this_file_path, '../../02_training/02_training/binADNI_weights_30_epochs__new_normalizer', params_destination)
  if not os.path.exists(params_destination_full_path):
    continue

  param_names = os.listdir(params_destination_full_path)
  param_names.sort()

  selected = None
  run = None
  runs = []
  trainings[params_destination] = runs

  # T1__bootstrap_index-01__initial_weights_index-01__001-tca-0.648-vca-0.643.pth
  for param_name in param_names:
    epoch = int(param_name[len('T1__bootstrap_index-01__initial_weights_index-01__'):len('T1__bootstrap_index-01__initial_weights_index-01__001')])
    
    # we use the last training epoch
    if fixed_epoch != None and epoch != fixed_epoch:
      continue

    bootstrap_index = int(param_name[len('T1__bootstrap_index-'):len('T1__bootstrap_index-01')])
    weights_index = int(param_name[len('T1__bootstrap_index-01__initial_weights_index-'):len('T1__bootstrap_index-01__initial_weights_index-01')])
    if run is None or run['bootstrap_index'] != bootstrap_index or run['weights_index'] != weights_index:
      selected = []
      run = {
        'bootstrap_index': bootstrap_index,
        'weights_index': weights_index,
        'selected': selected, 
      }
      runs.append(run)
    
    epoch_result = {
      'bootstrap_index': int(param_name[len('T1__bootstrap_index-'):len('T1__bootstrap_index-') + 2]),
      'initial_weights_index': int(param_name[len('T1__bootstrap_index-01__initial_weights_index-'):len('T1__bootstrap_index-01__initial_weights_index-') + 2]),
      'epoch': epoch,
      # 'training_loss': float(param_name[param_name.index('tcl-') + len('tcl-'):param_name.index('vcl-') - 1]),
      # 'validation_loss': float(param_name[param_name.index('vcl-') + len('vcl-'):param_name.index('tca-') - 1]),
      'training_accuracy': float(param_name[param_name.index('tca-') + len('tca-'):param_name.index('vca-') - 1]),
      'validation_accuracy': float(param_name[param_name.index('vca-') + len('vca-'):param_name.index('.pth') if min_relevance_inner_mask is None else param_name.index('tim--') - 1]),
      'training_relevance': None if min_relevance_inner_mask is None else float(param_name[param_name.index('tim--') + len('tim--'):param_name.index('vim--') - 1]),
      'validation_relevance': None if min_relevance_inner_mask is None else float(param_name[param_name.index('vim--') + len('vim--'):param_name.index('.pth')]),
      'file': param_name
    }

    if (min_training_accuracy is None or epoch_result['training_accuracy'] >= min_training_accuracy)\
        and (min_validation_accuracy is None or epoch_result['validation_accuracy'] >= min_validation_accuracy)\
        and (min_relevance_inner_mask is None or epoch_result['validation_relevance'] >= min_relevance_inner_mask):
      selected.append(epoch_result)

      # search for highest validation accuracy
      if params_destination not in trainings_highest or trainings_highest[params_destination]['validation_accuracy'] <= epoch_result['validation_accuracy']:
        trainings_highest[params_destination] = epoch_result

with open('trainings_filtered.json', 'w') as outfile:
  json.dump(trainings, outfile, indent=2)

with open('trainings_highest.json', 'w') as outfile:
  json.dump(trainings_highest, outfile, indent=2)