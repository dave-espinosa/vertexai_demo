# train.py


# ------------------------------- Libraries -----------------------------------

import ast
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("Numpy version: ", np.__version__)
print("Pandas version: ", pd.__version__)
print("Tensoflow version: ", tf.__version__)

# ------------------------ User-defined functions -----------------------------

def train_val_test_df_split(df, tr_s, vl_s, ts_s, v=False):
    
    assert_1 = "'tr_s' + 'vl_s' + 'ts_s' must be equal to 1"
    assert_2 = "'value_count()' on 'df['program_grouping']' must be larger than 3 for all samples"
    
    assert tr_s + vl_s + ts_s - 1 < 1e-15, assert_1

    tt_s = vl_s + ts_s # temporal for first splitting
    
    pgno = df.program_grouping.value_counts()>2
    
    assert len(pgno[~pgno].index.tolist()) < 1, assert_2

    # Initial train and test split
    train_df, test_df = train_test_split(
        df.copy(),
        test_size=tt_s,
        stratify=df["program_grouping"].values,
    )

    # Splitting the test set further into validation
    # and new test sets.
    val_df = test_df.sample(frac=vl_s/tt_s)
    test_df.drop(val_df.index, inplace=True)

    if v:
        print(f"Number of rows in training set: {len(train_df)}")
        print(f"Number of rows in validation set: {len(val_df)}")
        print(f"Number of rows in test set: {len(test_df)}")
    
    return train_df, val_df, test_df


def make_dataset(df, is_train=True):
    batch_size = 128
    labels = tf.expand_dims(tf.convert_to_tensor(df["program_grouping"].values), axis=-1)
    label_binarized = progroup_vectorizer_layer(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (   # Multiple inputs are required, this satisfies it
                tf.ragged.constant(df["new_skills"].values),
                df["job_title"],
            ),
            label_binarized
        )
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)    


print("\nUser-defined functions loaded and ready for use")


# ---------------------------- Data Importation -------------------------------

# Importing original dataframe
# Bucket path, Pandas can read directly from GCS!
path = "gs://onementor-ml-data/program_grouping_data.zip"
df = pd.read_csv(path, compression='zip')
df['new_skills'] = df['new_skills'].apply(ast.literal_eval)

print("\nData loaded and ready for use (shape {})".format(df.shape))


# ------------------------------ Data Cleaning --------------------------------

# Does not exist, this dataset is already clean.


# ------------------------- Corpus Pre - processing ---------------------------

print("\nCorpus pre-processing")
print("\tInputs")

# 'new_skills' vectorization layer preset
slol = df.new_skills.tolist()
skills_vocabulary = list(set([item.lower() for sublist in slol for item in sublist]))
len_skills_vocabulary = len(skills_vocabulary)
print(f"\t\tThere are {len(skills_vocabulary)} different terms in the dictionary of 'new_skills'")
skills_vocabulary = tf.data.Dataset.from_tensor_slices(skills_vocabulary)
skill_vectorizer_layer = tf.keras.layers.TextVectorization(
    split=None,
    ragged=True,
    name="skill_vectorizer_layer"
)
with tf.device("/GPU:0"):
    skill_vectorizer_layer.adapt(skills_vocabulary.batch(64))

# 'job_title' vectorization layer preset
jtl = df.job_title.tolist()
jobtitle_vocabulary = list(set([item.lower() for item in jtl]))
max_len_jobtitle_str = max([len(item) for item in jobtitle_vocabulary])
len_jobtitle_vocabulary = len(jobtitle_vocabulary)
print(f"\t\tThere are {len_jobtitle_vocabulary} different terms in the dictionary of 'job_title'")
jobtitle_vocabulary = tf.data.Dataset.from_tensor_slices(jobtitle_vocabulary)
jobtitle_vectorizer_layer = tf.keras.layers.TextVectorization(
    max_tokens=int(1.1*len_jobtitle_vocabulary),
    output_sequence_length=int(1.1*max_len_jobtitle_str),
    pad_to_max_tokens=True, # 'max_tokens' must be set
    name="jobtitle_vectorizer_layer"
)
with tf.device("/GPU:0"):
    jobtitle_vectorizer_layer.adapt(jobtitle_vocabulary.batch(64))

print("\tOutputs")

# 'program_grouping' vectorization layer preset
pgl = df.program_grouping.tolist()
progroup_vocabulary = list(set([item.lower() for item in pgl]))
max_len_progroup_str = max([len(item) for item in progroup_vocabulary])
len_progroup_vocabulary = len(progroup_vocabulary)
print(f"\t\tThere are {len_progroup_vocabulary} different terms in the dictionary of 'program_grouping'")
progroup_vocabulary = tf.data.Dataset.from_tensor_slices(progroup_vocabulary)
progroup_vectorizer_layer = tf.keras.layers.TextVectorization(
    output_mode='multi_hot',
    split=None
)
with tf.device("/GPU:0"):
    progroup_vectorizer_layer.adapt(progroup_vocabulary.batch(64))

# VOCABULARY TO MAKE PREDICTIONS: DO NOT ERASE
vocab = progroup_vectorizer_layer.get_vocabulary()


# ------------------------------ Data Splitting -------------------------------

print("\nData splitting, report:")
tr_s, vl_s, ts_s = 0.9, 0.075, 0.025
train_df, val_df, test_df = train_val_test_df_split(df, tr_s, vl_s, ts_s, v=True)
train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)


# -------------------------------- Build model --------------------------------

print("\nBuilding and compiling model... ", end="")

EMBEDDING_SIZE = 128

# 'new_skills' input branch (x1)
input_new_skills = tf.keras.layers.Input(shape=(None,), dtype=tf.string, ragged=True, name="skill_input")
x1 = skill_vectorizer_layer(input_new_skills)
x1 = tf.keras.layers.Embedding(skill_vectorizer_layer.vocabulary_size(), EMBEDDING_SIZE, name="skill_embedding")(x1)
x1 = tf.keras.layers.GlobalAveragePooling1D(name="skill_average")(x1) 

# 'job_title' input branch(x2)
input_job_titles = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="jobtitle_input")  #  ragged=True
x2 = jobtitle_vectorizer_layer(input_job_titles)
x2 = tf.keras.layers.Embedding(jobtitle_vectorizer_layer.vocabulary_size(), EMBEDDING_SIZE, name="jobtitle_embedding")(x2)
x2 = tf.keras.layers.GlobalAveragePooling1D(name="jobtitle_average")(x2)

# 'concatenation' of 'x1' & 'x2'
concat = tf.keras.layers.concatenate([x1, x2], axis=-1, name="concatenate")

# Deeper layers
x = tf.keras.layers.Dense(4*EMBEDDING_SIZE, activation='relu', name="dense1")(concat)
x = tf.keras.layers.Dense(EMBEDDING_SIZE, activation='relu', name="dense2")(x)
outputs = tf.keras.layers.Dense(progroup_vectorizer_layer.vocabulary_size(), activation="sigmoid", name="output")(x)
model = tf.keras.Model(inputs=[input_new_skills, input_job_titles], outputs=outputs)

# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])

print("Done!")

# -------------------------------- Train model --------------------------------

print("Training model... ", end="")

epochs = 10

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
)

print("Done!")

# Export model and save to GCS
BUCKET = 'gs://mentor-pilot-project-bucket'
FOLDER = '/mpg/model_dave'
model.save(BUCKET + FOLDER)

print("Cloud Storage destination: '{}'".format(BUCKET + FOLDER))
