#step 1 separate train and test data into train, validation, and test data

#gal_df 
#gftp_df 
#gaucfmc_df 
#gauc_df

def get_x_y(df):
    columns = df.columns
    x_column_name = [x for x in columns if x not in ["code", "species"]]
    x = df[x_column_name].values
    y = df["species"].values
    y_p = np.array(replace_strings(y))
    x = np.array(x)
    return x, y_p
    
x,y_p = get_x_y(gaucfmc_df)

y_p

import random

def permute_lists(list1, list2):
    # Generate a random permutation index
    permutation_index = list(range(len(list1)))
    random.shuffle(permutation_index)

    # Apply the same permutation to both lists
    permuted_list1 = [list1[i] for i in permutation_index]
    permuted_list2 = [list2[i] for i in permutation_index]

    return permuted_list1, permuted_list2
    
def partition_lists(list1, list2, fraction):
    if fraction >= 1:
        raise ValueError("Fraction must be less than 1")
    if fraction  < 0:
        raise ValueError("Fraction must be more than 0")
        
    num_elements = len(list1)
    num_selected = int(num_elements * fraction)

    # Partition elements based on the fraction
    selected_list1 = list1[:num_selected]
    selected_list2 = list2[:num_selected]
    remaining_list1 = list1[num_selected:]
    remaining_list2 = list2[num_selected:]

    return selected_list1, selected_list2, remaining_list1, remaining_list2

def get_training_and_test_data(features, labels, fraction):
    permutated_features, permuted_labels = permute_lists(features, labels)
    training_features, training_labels, test_features, test_labels = partition_lists(permutated_features, permuted_labels, fraction)
    training_features, training_labels, test_features, test_labels = np.array(training_features),  np.array(training_labels),  np.array(test_features),  np.array(test_labels)
    return training_features, training_labels, test_features, test_labels

x_train, y_train,x_test, y_test  = get_training_and_test_data(x, y_p, 0.5)

print(x_train, y_train)


def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""  
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()

print("Loaded the plot_curve function.")

#step 2 define model, training and evaluation, plotting functions

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# -------------------------------
# Model creation
# -------------------------------
def create_model(my_learning_rate, x_train, y_train):
    """Create and compile a deep neural net.
       Automatically infers input shape from x_train
       and number of classes from y_train.
    """

    # --- Infer input shape ---
    if isinstance(x_train, pd.DataFrame):
        input_shape = (x_train.shape[1],)
    elif isinstance(x_train, np.ndarray):
        if len(x_train.shape) == 1:
            input_shape = (1,)
        else:
            input_shape = (x_train.shape[1],)
    else:
        raise TypeError("x_train must be a pandas DataFrame or numpy array.")

    # --- Infer number of classes ---
    if isinstance(y_train, (pd.Series, pd.DataFrame)):
        num_classes = len(np.unique(y_train.values))
    else:
        num_classes = len(np.unique(y_train))

    if num_classes < 2:
        raise ValueError("y_train must contain at least 2 classes.")

    # --- Build model ---
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    # Hidden layers
    model.add(tf.keras.layers.Dense(units=20, activation="relu"))
    model.add(tf.keras.layers.Dense(units=20, activation="relu"))
    model.add(tf.keras.layers.Dense(units=20, activation="relu"))

    # Dropout layer
    model.add(tf.keras.layers.Dropout(rate=0.2))

    # Output layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation="softmax"))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -------------------------------
# Training function
# -------------------------------
def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    """Train the model with robust preprocessing and error checks.
       Returns (epochs, hist, encoder) where encoder may be None.
    """

    # Convert to numpy arrays if pandas
    if isinstance(train_features, pd.DataFrame):
        train_features = train_features.values
    if isinstance(train_label, (pd.Series, pd.DataFrame)):
        train_label = train_label.values.ravel()

    # Encode labels if they are strings
    encoder = None
    if train_label.dtype.kind in {"U", "S", "O"}:  # unicode, string, object
        encoder = LabelEncoder()
        train_label = encoder.fit_transform(train_label)

    # Ensure numeric types
    train_features = np.array(train_features, dtype=np.float32)
    train_label = np.array(train_label, dtype=np.int32)

    # Sanity checks
    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels shape: {train_label.shape}")
    print(f"Classes in labels: {np.unique(train_label)}")

    # Train model
    history = model.fit(
        x=train_features,
        y=train_label,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=validation_split,
        verbose=1,
    )

    epochs_out = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs_out, hist, encoder


# -------------------------------
# Evaluation
# -------------------------------
def evaluate_model(model, test_features, test_label, encoder=None, batch_size=None):
    """Evaluate the model, encoding labels if needed."""
    if isinstance(test_features, pd.DataFrame):
        test_features = test_features.values
    if isinstance(test_label, (pd.Series, pd.DataFrame)):
        test_label = test_label.values.ravel()

    if encoder is not None:
        test_label = encoder.transform(test_label)

    test_features = np.array(test_features, dtype=np.float32)
    test_label = np.array(test_label, dtype=np.int32)

    return model.evaluate(x=test_features, y=test_label, batch_size=batch_size, verbose=1)


# -------------------------------
# Plotting
# -------------------------------
def plot_curve(epochs, hist, list_of_metrics_to_plot):
    """Plot metrics vs epochs with safety checks."""
    plt.figure(figsize=(8, 6))
    for m in list_of_metrics_to_plot:
        if m in hist:
            plt.plot(epochs, hist[m], label=m)
        else:
            print(f"⚠️ Warning: metric '{m}' not in history")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()


#step 3 Execute training

# -------------------------------
# Hyperparameters
# -------------------------------
learning_rate = 0.008
epochs = 150
batch_size = 3
validation_split = 0.3

# -------------------------------
# Build and train model
# -------------------------------
try:
    my_model = create_model(learning_rate, x_train=x_train, y_train=y_train)
except Exception as e:
    raise RuntimeError(f"❌ Failed to create model: {e}")

try:
    epochs_run, hist, encoder = train_model(
        my_model,
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )
except Exception as e:
    raise RuntimeError(f"❌ Training failed: {e}")

# -------------------------------
# Plot metrics
# -------------------------------
list_of_metrics_to_plot = ["accuracy"]
if hist is not None and not hist.empty:
    plot_curve(epochs_run, hist, list_of_metrics_to_plot)
else:
    print("⚠️ Warning: Training history is empty, skipping plot.")

# -------------------------------
# Evaluate on test set
# -------------------------------
print("\n🔍 Evaluating model against the test set...")
try:
    test_loss, test_acc = evaluate_model(
        my_model,
        x_test,
        y_test,
        encoder=encoder,
        batch_size=batch_size
    )
    print(f"✅ Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
	
	
#step 4 Create Statistics dataframe for metric