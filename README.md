# ExIM: Exploring Intent of Music Listening for Retrieving User-generated Playlists

This repository provides the data and dashboard to showcase the work of our paper accepted by CHIIR 25. 

## Abstract

Music psychology has identified various reasons why people listen to music, based on empirical evidence from interviews and surveys.
In this paper, we take a data-driven approach that adopts both pre-trained Sentence Transformers and Cross Encoder, as well as
graph-based clustering to first determine music listening intents and then explore user-generated playlists by comparing the title
to the listening intents. For this purpose, we first investigated whether 129 established listening functions, previously identified
by Schäfer et al. (2013), could be meaningfully clustered into broader listening intents. While Schäfer et al. (2013) introduced three
broad dimensions of music listening, this work aimed to identify smaller, context-specific intents to capture more nuanced intents.
The resulting clusters were then evaluated through a first survey to select the clusters of the best performing model. In a second
survey, music listening intent clusters were explored in more detail to obtain a deeper understanding of their significance for music
retrieval and recommendation. Lastly, the playlist selection per intent and characteristics of listening with intent were further explored
through a third survey. Given the encouraging results of the evaluation of the computed clusters (92% of clusters judged consistent by
participants) and the insight that most (> 50%) of the participants search for playlists for a specific intent, we propose a browsing
system that categorizes playlists based on their intent and enables users to explore similar playlists. Our approach is further visualized
in a dashboard to explore and browse through playlists in intent space.

## Data

The data can be found in the folder **/data** and consists of the following files:

- intent_data.json: All intents with descriptive information.
	- intent_id (ID of intent)
	- intent_name (Name of Intent)
	- main_listening_function (listening function which has the highest cosine similarity to all listening functions)
	- listening_functions (All listening functions in intent cluster)
	- listening_function_factors (Mean selection rate for each listening function to fit the intent)
	- survey_intent_names (Intent names given by the participants of the survey)
	
- intent_to_characteristics.json: 
	- intent_id (ID of intent) 
	- intent_name (Name of Intent) 
	- Musical features (Mean Rating for selecting songs based on musical features when listening with intent: 1 = Strongly disagree, 5 = Strongly agree) 
	- Specific playlist (Mean Rating for listening to specific playlist when listening with intent: 1 = Strongly disagree, 5 = Strongly agree)
	- Personal connection (Mean Rating for selecting songs based on personal connection when listening with intent: 1 = Strongly disagree, 5 = Strongly agree) 
	- Normally listened songs (Mean Rating for listening to normally listened songs when listening with intent: 1 = Strongly disagree, 5 = Strongly agree)  
	- Alltime favorites (Mean Rating for selecting alltime favorite songs when listening with intent: 1 = Strongly disagree, 5 = Strongly agree) 
	- Current favorites (Mean Rating for selecting current favorite songs when listening with intent: 1 = Strongly disagree, 5 = Strongly agree) 
	- Specific songs (Mean Rating for selecting specific songs when listening with intent: 1 = Strongly disagree, 5 = Strongly agree) 
	- mean_listen_frequency (Mean listening frequency of participants listening to intent: 0 = Never, 1 = Rarely, 2 = Sometimes, 3 = Often, 4 = Always)

## Scripts

Coming soon...


## Run Dashboard

- Install requirements from "requirements.txt"
- Run "python /dashboard/app.py"
- Open "http://127.0.0.1:8050/exploring_intent"

## Citation

Coming soon...

## Copyright

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
