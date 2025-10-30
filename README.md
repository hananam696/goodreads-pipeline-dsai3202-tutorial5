# goodreads-pipeline-dsai3202-tutorial5

# HOMEWORK PART -1
The dataset contains 14 columns and around 20,000 rows, consisting of string, integer, and double data types. From the column statistics, most columns have consistent and valid data. The “publication_year” column has about 13% missing values, while all other columns have 0% missing data. This indicates that the dataset is mostly complete and reliable, with only one field requiring attention. The data quality rule was scheduled to run every Sunday within a 12-hour window to monitor missing values and maintain data integrity. Overall, the dataset is clean and well-structured, requiring minimal preprocessing before feature engineering.

# Homework 2
### – Date Processing & Filtering

Started with Option 1 for creating the ISO date and filtering. After obtaining the output, the project was deleted to proceed with Option 2 in order to try both approaches.

Option 1 (Initial Attempt):

- Created an Athena query to convert the date_added column to ISO format.
- Created a table named curate_1_5.
- Imported the old recipes into a new project.
- Filtered the dates using the date_iso_flagged column, keeping only dates on or after January 1, 2014.

Note: This project was later deleted to attempt a different approach (Option 2).

Option 2 (Final Approach):

- Extracted the year, month, and day from the date_added column.
- Combined these values to create an ISO-formatted date.
- Filtered the dataset to keep only reviews from 2014 onwards.

*Result*: The dataset is now filtered to include only relevant reviews from 2014 and beyond.