import os
import music21 as m21
import json 
import keras
import numpy as np


KERN_DATASET_PATH = "deutschl/erk"
ACCEPTABLE_DURATIONS = [ 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "mapping.json"

#_________________________________________________________________________#
def load_songs_in_kern(dataset_path):
    #go through all the files in the dataset and load them with music21
    songs = []
    for path, _, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs
#_________________________________________________________________________#


#_________________________________________________________________________#
def has_acceptable_duration(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True
#_________________________________________________________________________#


#_________________________________________________________________________#
def transpose(song):
    # try to get the key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]  

    # estimate key using music 21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    
    # get the interval for transposition 
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C")) 
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A")) 

    #transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song
#_________________________________________________________________________#

#_________________________________________________________________________#
def encode_song(song, timestep = 0.25):
    encoded_song = []

    for event in song.flat.notesAndRests:
        
        # for notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        
        #for rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        
        #convert the note or rest into time series notation
        steps = int(event.duration.quarterLength / timestep)

        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    
    #cast the encoded song to a string
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song
#_________________________________________________________________________#


#_________________________________________________________________________#
def preprocess(dataset_path):
    # load the folk songs 
    print("Loading Songs")
    songs = load_songs_in_kern(dataset_path)
    print(f"loaded {len(songs)} songs")
     
    # filter out songs with non-acceptable duarations
    for index, song in enumerate(songs):
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time-series representation 
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(index))

        with open(save_path, "w") as fp:
            fp.write(encoded_song) 
#_____________________________________________________________________#


#_____________________________________________________________________#
def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song     
#_____________________________________________________________________#


#_____________________________________________________________________#
def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    # load encode songs and add delimiters
    new_song_delimiter = " / " * sequence_length
    songs = ''

    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + new_song_delimiter
    
    songs = songs[:-1]

    # save string that contains the whole dataset
    with open(file_dataset_path, 'w') as fp:
        fp.write(songs)

    return songs
#_____________________________________________________________________#


#_____________________________________________________________________#
def create_mapping(songs, mapping_path):
    mappings = {}
    
    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    for index, symbol in enumerate(vocabulary):
        mappings[symbol] = index
    
    # save vocabulary to a jsom file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)
#_____________________________________________________________________#


#_____________________________________________________________________#
def convert_songs_to_int(songs):
    int_songs = []

    # load the mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # cast songs string to a list
    songs = songs.split()


    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs
#_____________________________________________________________________#


def generate_training_sequences(sequence_length):
    inputs = []
    targets = []

    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    
    # generate the training sequences
    num_of_sequences = len(int_songs) - sequence_length
    for i in range(num_of_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences  
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes = vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

#_____________________________________________________________________#
def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
#_____________________________________________________________________#





if __name__ == "__main__":
    main()
   
    # song.show("midi")
    # transposed_song = transpose(song)
    # transposed_song.show("midi")