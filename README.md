# MiniLM_merdge_csv
Use mini embedded llm to handle a large .csv file counting duplicates in entities then merge them saving related info in first created item.

model name:
sentence-transformers/all-MiniLM-L6-v2

*This script is aimed at using a small model for the best performance, even on not powerful hardware*
All the main errors should be printed in the console
The current job status is also printed to the console

You should prepare your files to be placed in the same folder with this script.

OPEN THE SCRIPT TO SET UP:
1. Change the input file key 'YOUR_FILE.csv' to your actual main .csv file name
FILE_IN = 'YOUR_FILE.csv'
*you can change also the name of output file
FILE_OUT = 'duplicates_merged.csv'

2. In your main .csv file get the column names to replace the key of variables

#The name of the column 'FIRST_RECORD' by which we determine the first duplicate that was created
FIRST_RECORD = 'FIRST_RECORD'
#The name of a column 'COLUMN_ONE' that is specific to containing unique information about an object
COLUMN_ONE = 'COLUMN_ONE'
#The name of a column 'COLUMN_TWO' that is specific to containing unique information about an object
COLUMN_TWO = 'COLUMN_TWO'

3. You can change the similarity threshold for clustering. Can be configured, a higher number means stricter similarity requirements
SIMILARITY_THRESHOLD = 0.5

Run and enjoy a big piece of manual work of comparing big data is done pretty quick!
