import argparse
import sys
import traceback

# from icodemix_lang_ident_classifier_app.data_ops import process_bhasha_dataset
from icodemix_lang_ident_classifier_app.model_ops import model_ops_api


def main():
    parser = argparse.ArgumentParser(description="CLI Dispatcher")
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--operation_mode", type=str, required=True)

    args, unknown = parser.parse_known_args()

    operation = args.operation_mode

    try:
        # if operation == "process_bhasha_dataset":
        #     process_bhasha_dataset.main()

        # elif operation == "model_hyperparameter_optimization":
        if operation == "model_hyperparameter_optimization":
            model_ops_api.main()

        else:
            raise ValueError(f"Unknown operation_mode: {operation}")

    except Exception as e:
        print("Error:", e)
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()