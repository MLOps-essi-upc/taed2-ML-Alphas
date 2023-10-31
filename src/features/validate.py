#!pip install great_expectations --quiet
#!pip install datasets --quiet

#!great_expectations init

import great_expectations as ge
from datasets import load_dataset
import pandas as pd


context = ge.data_context.DataContext()
context.add_or_update_expectation_suite("alzheimer_training_suite")

datasource = context.sources.add_or_update_pandas(name="alzheimer_dataset")

dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
train = dataset.to_pandas()

data_asset = datasource.add_dataframe_asset(name="training", dataframe=train)


batch_request = data_asset.build_batch_request()
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="alzheimer_training_suite",
    datasource_name="alzheimer_dataset",
    data_asset_name="training",
)

validator.expect_table_columns_to_match_ordered_list(
    column_list=[
        "image",
        "label",
    ]
)

validator.expect_column_values_to_be_unique("image")

validator.expect_column_values_to_not_be_null("image")

validator.expect_column_values_to_not_be_null("label")

validator.expect_column_values_to_be_of_type("label", "int64")

validator.expect_column_values_to_be_between("label", min_value=0, max_value=3)

validator.save_expectation_suite(discard_failed_expectations=False)

checkpoint = context.add_or_update_checkpoint(
    name="my_checkpoint",
    validator=validator,
)

checkpoint_result = checkpoint.run()
context.view_validation_result(checkpoint_result)
