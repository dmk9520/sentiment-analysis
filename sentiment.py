import numpy as np
import tensorflow as tf
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


# shuffle data
combined_data = list(zip(reviews, labels))
random.shuffle(combined_data)
reviews, labels = zip(*combined_data)

# 2) Tokenization to encode the words in reviews
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(reviews)

# 5) applying pad sequences
sequences= tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# 6) Split data into train and test sets
train_size =int(0.8 * len(padded_sequences))
train_reviews, test_reviews = padded_sequences[:train_size], padded_sequences[train_size:]
train_labels, test_labels = np.array(labels[:train_size]), np.array(labels[train_size:])

# Model building 8) Defining the training architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 11) Training the model
model.fit(train_reviews, train_labels, epochs=25, batch_size=64, validation_split=0.1)

# 12) Model evaluation
test_loss, test_accuracy = model.evaluate(test_reviews, test_labels)
print(f'Test Accuracy: {test_accuracy}')

model.save('sentiment_model.h5')

#loading the trained model
loaded_model =tf.keras.models.load_model('sentiment_model.h5')

# making predictions
def predict_sentiment(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    return "Positive review" if prediction >= 0.5 else "Negative review"