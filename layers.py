from tensorflow.keras.models import load_model

# Load the model
model = load_model('action3GRU.h5')

# Print the summary of the model
model.summary()
