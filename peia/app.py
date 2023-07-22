from flask import Flask, render_template, request, redirect, url_for, session
import torch
from PEIA import load_model, CustomDataset
from torch.utils.data import DataLoader
import os
from tinytag import TinyTag
import pandas as pd
from settings import emotions, mother_tongue_mapping, batch_size, genre_song_count, genre_mapping, load_similar_songs

app = Flask(__name__)
app.secret_key = 'a1b2c3'

model = load_model()

def create_tensor(data, dtype, size):
    return torch.full((size,), data, dtype=dtype)

def load_user_mood_data(request_form, selected_genres):
    size = sum(genre_song_count[genre] for genre in selected_genres)
    user = {
        'age': create_tensor(float(request_form.get('age')), dtype=torch.float32, size=size),
        'gender': create_tensor(1.0 if request_form.get('gender') == 'male' else 0.0, dtype=torch.float32, size=size),
        'mother tongue': create_tensor(mother_tongue_mapping[request_form.get('country')], dtype=torch.int64, size=size)
    }

    mood = {
        'mood': create_tensor(float(request_form.get('mood')), dtype=torch.float32, size=size),
        **{emotion: create_tensor(1.0 if emotion in request_form.getlist('emotion') else 0.0, dtype=torch.float32, size=size) for emotion in emotions}
    }

    return user, mood

def load_genre_data(selected_genres):
    genre_data = {
        'genre1': [],
        'genre2': []
    }
    for genre in selected_genres:
        file_path = f'spected_music/{genre}_data.pt'
        if os.path.isfile(file_path):
            data = torch.load(file_path, map_location=torch.device('cpu'))
            genre_data['genre1'].append(data[:,0])
            genre_data['genre2'].append(data[:,1])
            
    genre_data['genre1'] = torch.cat(genre_data['genre1'])
    genre_data['genre2'] = torch.cat(genre_data['genre2'])

    return genre_data

def process_post_request(request_form):
    global model
    session['age'] = request_form.get('age')
    session['gender'] = request_form.get('gender')
    session['mother tongue'] = request_form.get('country')
    session['mood'] = request_form.get('mood')
    session['emotions'] = {emotion: (emotion in request_form.getlist('emotion')) for emotion in emotions}
    session['selected_genres'] = request.form.getlist('genre[]')
    
    selected_genres = session['selected_genres']

    user, mood = load_user_mood_data(request_form, selected_genres)
    Genre = load_genre_data(selected_genres)
    
    customdataset = CustomDataset(user, mood, Genre)
    dataloader = DataLoader(customdataset, batch_size=batch_size, shuffle=False)

    outputs = []
    with torch.no_grad():
        for data in dataloader:
            output, _, _ = model(data['user'], data['mood'], data['Genre'])
            outputs.append(output)
    all_outputs = torch.cat(outputs).flatten()
    
    topk_values, topk_indices = torch.topk(all_outputs, 15)
    max_indices = topk_indices[topk_values == topk_values.max()]
    if len(max_indices) <= 5:
        selected_indices = topk_indices[:5]
    else:
        shuffled_indices = max_indices[torch.randperm(max_indices.size(0))]
        selected_indices = shuffled_indices[:5]
    
    top5_music_files = []
    
    for index in selected_indices.tolist():
        output_counter = 0
        for genre in selected_genres:
            if index<output_counter+genre_song_count[genre]:
                index -= output_counter
                top5_music_files.append(f"emotifymusic/{genre}/{index+1}.mp3")
                break
            output_counter+=genre_song_count[genre]
    top5_music_files_str = ','.join(top5_music_files)
    return top5_music_files_str

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        top5_music_files_str = process_post_request(request.form)
        return redirect(url_for('result', top5_songs=top5_music_files_str))

    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        feedback_info = {
            'age': int(session.get('age')),
            'gender': 1 if session.get('gender') == 'male' else 0,
            'mother tongue': session.get('mother tongue'),
            'mood': int(session.get('mood')),
            **{emotion: int(session.get('emotions')[emotion]) for emotion in emotions}
        }

        excel_file_path = 'feedback.xlsx'
        feedback_df = pd.read_excel(excel_file_path) if os.path.isfile(excel_file_path) else pd.DataFrame()
        
        new_feedback = []
        for i in range(5):
            feedback = request.form.get(f'feedback{i+1}')
            if feedback is None:
                continue
            
            song_id = int(request.form.get(f'track id{i+1}'))
            song_genre = request.form.get(f'genre{i+1}')
            song_feedback = {
                'track id': song_id,
                'genre': song_genre,
                'liked': int(request.form.get(f'feedback{i+1}')),
                **feedback_info
            }

            new_feedback.append(song_feedback)
        
        new_feedback_df = pd.DataFrame(new_feedback)
        feedback_df = pd.concat([feedback_df, new_feedback_df], ignore_index=True)
        feedback_df.to_excel(excel_file_path, index=False)
        session.clear()
        return redirect(url_for('home'))


    top5_songs = request.args.get('top5_songs').split(',')
    music_with_similar_songs = []
    
    classical_top_5_similar = load_similar_songs('classical_top_5_similar.pkl')
    rock_top_5_similar = load_similar_songs('rock_top_5_similar.pkl')
    pop_top_5_similar = load_similar_songs('pop_top_5_similar.pkl')
    electronic_top_5_similar = load_similar_songs('electronic_top_5_similar.pkl')
    
    for song in top5_songs:

        song_path = os.path.join('static/', song)
        tag = TinyTag.get(song_path)
        song_title = tag.title
        song_index = int(song.split('/')[-1].split('.')[0]) - 1
        song_genre = song.split('/')[-2]
        
        if song_genre == 'classical':
            top_5_similar_songs = classical_top_5_similar
        elif song_genre == 'rock':
            top_5_similar_songs = rock_top_5_similar
        elif song_genre == 'pop':
            top_5_similar_songs = pop_top_5_similar
        elif song_genre == 'electronic':
            top_5_similar_songs = electronic_top_5_similar
            
        similar_song_indices_and_genres = top_5_similar_songs[song_index]
        similar_song_files = []
        similar_song_titles = []
        
        for index_and_genre in similar_song_indices_and_genres:
            index = round(index_and_genre)
            genre = genre_mapping[round((index_and_genre % 1) * 10)]
            song_file = f"emotifymusic/{genre}/{index+1}.mp3"
            similar_song_files.append(song_file)
            similar_song_titles.append(TinyTag.get(os.path.join('static/', song_file)).title)

        music_with_similar_songs.append((song, song_title, list(zip(similar_song_files, similar_song_titles))))

    return render_template('result.html', music=music_with_similar_songs)

if __name__ == "__main__":
    app.run(debug=True)