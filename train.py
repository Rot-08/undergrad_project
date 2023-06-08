from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import keras


OUTPUT_UNITS = 38
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 1
BATCH_SIZE = 64
MODEL_PATH = "Melody-Generator.h5"



if __name__ == "__main__":

    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    model = keras.models.load_model(MODEL_PATH)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(MODEL_PATH)   
