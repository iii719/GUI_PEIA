import pickle
import os

batch_size=16
pred_emotions = ['pred_amazement', 'pred_solemnity', 'pred_tenderness', 'pred_nostalgia', 'pred_calmness', 'pred_power', 'pred_joyful_activation', 'pred_tension', 'pred_sadness']
emotions = ['amazement', 'solemnity', 'tenderness', 'nostalgia', 'calmness', 'power', 'joyful_activation', 'tension', 'sadness']
mother_tongue_mapping = {
    "Arabic": 0,
    "Bengali": 1,
    "Catalan": 2,
    "Chinese": 3,
    "Croatian": 4,
    "Czech": 5,
    "Danish": 6,
    "Dutch": 7,
    "English": 8,
    "Estonian": 9,
    "Farsi": 10,
    "Finnish": 11,
    "French": 12,
    "German": 13,
    "Greek": 14,
    "Hebrew": 15,
    "Hindi": 16,
    "Hungarian": 17,
    "Icelandic": 18,
    "Italian": 19,
    "Japanese": 20,
    "Korean": 21,
    "Latvian": 22,
    "Macedonian": 23,
    "Malay": 24,
    "Norwegian": 25,
    "Polish": 26,
    "Portuguese": 27,
    "Romanian": 28,
    "Russian": 29,
    "Serbian": 30,
    "Slovenian": 31,
    "Spanish": 32,
    "Swedish": 33,
    "Tamil": 34,
    "Turkish": 35,
    "Ucrainian": 36,
    "Urdu": 37
}

def load_similar_songs(filepath):
    with open('static/'+filepath, 'rb') as f:
        top_5_similar_songs = pickle.load(f)
    return top_5_similar_songs


genre_song_count = {}

genre_dirs = ['classical', 'rock', 'pop', 'electronic']
base_dir = 'static/emotifymusic'

for genre in genre_dirs:
    genre_dir = os.path.join(base_dir, genre)
    files_in_dir = os.listdir(genre_dir)
    files_in_dir = [int(os.path.splitext(file)[0]) for file in files_in_dir]
    files_in_dir = sorted(files_in_dir)
    last_file = files_in_dir[-1]
    genre_song_count[genre] = last_file
    
genre_mapping = {
            0: 'classical',
            1: 'rock',
            2: 'pop',
            3: 'electronic'
        }

genre_mapping_files = {
        'classical': 'classical_top_5_similar.pkl',
        'rock': 'rock_top_5_similar.pkl',
        'pop': 'pop_top_5_similar.pkl',
        'electronic': 'electronic_top_5_similar.pkl'
    }