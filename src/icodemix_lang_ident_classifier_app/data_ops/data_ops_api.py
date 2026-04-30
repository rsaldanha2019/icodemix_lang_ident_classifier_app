# -*- coding: utf-8 -*-
# ---------------------------------------------------------
# @File             : data_ops_api.py
# @Time             : 2026-04-24 10:04
# @CodeCheck        : 
# 14 11 12 11 42 15 14 54 23 32 34 42 21 11 23 33 11 14 31 11 43 14 42 11 23 13 24 42
# Contact the author through email on the github README
# if you intend to use this package for commercial purposes
# ---------------------------------------------------------

import argparse

from icodemix_lang_ident_classifier.api.process_bhasha_dataset import ProcessBhashaDataset
from icodemix_lang_ident_classifier.language.utils.log_utils import LogUtils
from icodemix_lang_ident_classifier.language.utils.property_utils import PropertyUtils

class DataOps:
    def __init__(self, args, log, props):
        self.args = args
        self.log = log
        self.props = props

def main():
    try:
        parser = argparse.ArgumentParser(description="Process Bhasha Abhijnaanam Dataset")
        parser.add_argument(
            "--config_file_path",
            type=str,
            required=True,
            help="Pass the yaml config file path",
        )
        parser.add_argument(
            "--operation_mode",
            type=str,
            required=True,
            help="Bame of the data operation mode",
        )
        args, unknown_args = parser.parse_known_args()
        props = PropertyUtils().get_yaml_config_properties(config_file=args.config_file_path)
        log = LogUtils().get_time_rotated_log(props)
        log.info(f"Unknown args {unknown_args} hence ignored")
        if args.operation_mode not in ["process_bhasha_dataset"]:
            raise ValueError("Only operation_mode values allowed are process_bhasha_dataset")
        DataOps(args=args, log=log, props=props)

        if args.operation_mode == "process_bhasha_dataset":
          ProcessBhashaDataset(log=log, props=props)  
    except argparse.ArgumentError as e:
        print(f"Error: {e}")
        parser.print_help()

if __name__ == "__main__":
    main()