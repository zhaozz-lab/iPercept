import itertools
import logging
import pickle

from densebag_utils import get_average_df_from_files, get_all_submission_files, \
    calculate_mse, get_angular_error, load_validation_gaze

logger = logging.getLogger(__name__)




def get_result_for_B(B, df_true, submission_files):
    print('>>> B: ', B)
    combos = list(itertools.combinations(submission_files, B))[:100]

    list_mse = [calculate_mse(get_average_df_from_files(combo), df_true) for combo in combos]
    list_angular = [get_angular_error(get_average_df_from_files(combo), df_true) for combo in combos]
    list_pred = [get_average_df_from_files(combo) for combo in combos]

    results_for_B = {
        'mse': list_mse,
        'angular': list_angular,
        'pred': list_pred
    }
    return results_for_B


if __name__ == "__main__":
    base_path = "../../outputs/DenseBagValidation/"
    model_prefix = "DenseBag_Validation_RS"
    file_prefix = "validation_predictions"
    df_true = load_validation_gaze()
    submission_files = get_all_submission_files(base_path, model_prefix, file_prefix)

    results = {B: get_result_for_B(B, df_true, submission_files) for B in range(1, len(submission_files) -1 )}
    print(results.keys())


    path_out = '../../outputs/DenseBagValidation/validation_data.pickle'
    with open(path_out, 'wb') as out_file:
        pickle.dump(results, out_file)


    # with open(path_out, 'rb') as in_file:
    #     d = pickle.load(in_file)
    #     print(d.keys())
    #     print(d[1])

# {B: {mse:[], angular: [], pred:[(pitch, yaw)]}}
# i.e:
# b = {1: {'mse': [0.005, 0.003, 0.006], 'angular': [5.5, 5.3, 5.6], 'pred': [(-0.02, 0.01), (-0.02, 0.02), (-0.03, -0.01)]},
#     2: {'mse': [0.005, 0.003, 0.006], 'angular': [5.5, 5.3, 5.6], 'pred': [(-0.02, 0.01), (-0.02, 0.02), (-0.03, -0.01)]}
#      }