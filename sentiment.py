import os
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#defining the base directory
base_dir= 'C:/Users/msi/Downloads/sorted_data_acl/sorted_data_acl'

# directories for different domains
domains =['books', 'dvd', 'electronics', 'kitchen_&_housewares']

#empty lists to hold reviews and labels
reviews = []
labels =[]

for domain in domains:
    #file paths using os.path.join
    positive_path = os.path.join(base_dir,domain, 'positive.review')
    negative_path = os.path.join(base_dir, domain,'negative.review')
