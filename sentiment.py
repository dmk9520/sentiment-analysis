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

    #checking the file paths
    if os.path.exists(positive_path) and os.path.exists(negative_path):
        try:
            with open(positive_path, 'r', encoding='utf-8') as pos_file:
                pos_reviews = pos_file.readlines()
                reviews.extend(pos_reviews)            # 3) Encoding the words for positive and negative reviews
                labels.extend([1] * len(pos_reviews))  # Assigning label 1 for positive reviews
            
            with open(negative_path, 'r', encoding='utf-8') as neg_file:
                neg_reviews = neg_file.readlines()
                reviews.extend(neg_reviews)
                labels.extend([0] * len(neg_reviews))  # assign label 0 for negative reviews
        
        except Exception as e:
            print(f"Error reading files in {domain} domain: {e}")

