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

# Homework 3
### – Feature Engineering

This part focuses on creating new features through AWS DataBrew. Two new features, namely user_avg_rating, which gives the mean of the rating column, and rating_deviation, which calculates how much each individual rating differs from the user’s average rating, were created using the Group By function in AWS. Furthermore, three more aggregated columns were created:

user_rating_std: It calculates the standard deviation of ratings per user to understand how consistent or varied each user’s ratings are.

user_total_reviews: It counts the total number of reviews made by each user.

ratings_count_log: It represents book popularity and was created by first adding one to the ratings count and then applying the natural logarithm (LN) function to this added column.

Result: At the end of this process, the dataset contained five engineered features: rating_deviation, user_avg_rating, user_rating_std, user_total_reviews, and ratings_count_log.

# Homework 4
### - Final Part, processing jobs on full dataset

In this stage, I ran the full feature extraction job in Amazon SageMaker to generate advanced features from the Goodreads dataset. First, I performed a trial run on about 2000 rows to ensure the script worked correctly, and it successfully produced an output file in S3. Then, I tested on 500 rows to confirm that all functions were running properly. After that, I executed the full dataset by disabling two heavy functions (zero-shot classification and DistilBERT embeddings) to reduce cost and processing time due to limited credits, and it successfully completed with the remaining features. Finally, I attempted to run the full dataset again with all functions enabled using six ml.m5.xlarge instances and 120 GB of storage, but after running for more than 10 hours, the credits were exceeded before the job could finish.

### - Final Features Selection:
I decided to keep all the newly engineered text features along with the key identifiers — book_id, user_id, and review_id. These features capture the important linguistic, structural, and behavioral aspects of the reviews. I removed 'review_text','review_length_raw' and then disabled zero-shot classification and DistilBERT embedding features to reduce computational cost and make the processing job faster, since these two steps require more memory and GPU resources. (The code for this is given in under the folder documented code)
