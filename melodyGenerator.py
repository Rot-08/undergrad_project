import keras
import json
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import numpy as np
import music21 as m21


class MelodyGenerator:

    def __init__(self, model_path = "Melody-Generator.h5"):
        self.model_path = model_path
        self.model  = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self.mappings = json.load(fp)

        self.start_symbols = ['/'] * SEQUENCE_LENGTH

    def sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature

        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        
        choices = range(len(probabilities))

        index = np.random.choice(choices, p=probabilities)

        return index


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        
        # create seed from start symbols
        seed = seed.split()
        melody = seed
        seed = self.start_symbols + seed
        
        # map seed to integers
        seed = [self.mappings[symbol] for symbol in seed]

        for  _ in range(num_steps):
            # limit the max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            one_hot_seed = keras.utils.to_categorical(seed, num_classes= len(self.mappings))

            one_hot_seed = one_hot_seed[np.newaxis, ...]

            probabilities = self.model.predict(one_hot_seed)[0]

            output_int = self.sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to the encoding
            output_symbol = [k for k, v in self.mappings.items() if v == output_int][0]

            # check if this is the end of the melody
            if output_symbol == '/':
                break
            melody.append(output_symbol)
        
        return melody


    def save_melody(self, melody, format = "midi", file_name = "mel.mid", step_duration = 0.25):
        
        # create a music21 stream
        stream =  m21.stream.Stream()
    
        # parse all the symbols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1


        for index, symbol in enumerate(melody):
            # handle notes/rests
            if (symbol != "_") or (index +1 == len(melody)):
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    
                    # rest
                    if start_symbol == "r":
                        m21.event = m21.note.Rest(quarterLength = quarter_length_duration)

                    #note
                    else:
                        m21.event = m21.note.Note(int(start_symbol), quarterLength = quarter_length_duration)


                    stream.append(m21.event)
                    step_counter = 1
                
                start_symbol = symbol


            # handle prolongation sign
            else:
                step_counter += 1


        # write the m21 stream to a midi file
        stream.write(format, file_name)




if __name__ == "__main__":
    mg = MelodyGenerator()

    seed = "67 _ _ _ _ _ 67 _ 69 _ _ _ 67 _ _ _ 67 _ _ _ _ _ 67 _ 65 _ _ _ 64 _ _ _ 67 _ _ _ _ _ 67 _ 65 _ _ _ 64 _ _ _ 62 _ _ _ _ _ _ _ 60 _ _ _ _ _ _ _ 67 _ _ _ _ _ 67 _ 69 _ _ _ 67 _ _ _ 67 _ _ _ _ _ 67 _ 65 _ _ _ 64 _ _ _ 67 _ _ _ _ _ "
    melody  = mg.generate_melody(seed, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature= 0.7)
    print(melody)

    mg.save_melody(melody)