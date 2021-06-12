"""
  `preproces.py`
  
  Loads two datasets collected from the Spotify platform and try to link their records
  This script reads and preprocesses these two datasets:
  (D1) Spotify's Audio Features dataset, ~600k tracks from 1922-2021 (tracks.csv)
      https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks
  (D2) Spotify's Worldwide Daily Song Ranking dataset, 2017-2018 (data.csv)
      https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking

  The link is done is this way:
  1. Dataset D2 contains records of top streamed tracks, so answers to queries like "What is the
     name/artist of the N most streamed tracks during Jun 2018 in Brazil?" can be given.
  2. Dataset D1 describes audio features of tracks, and each track record is combined with
     a "song name" name and "group of artists" (e.g., 'Bom Rapaz' of 'Fernando & Sorocaba', featuring
     'Jorge & Mateus').
  3. The relationship between these two datasets is created based on similarity of the song name and artists
     combined with each track. To this purpose, several strategies are pursued.

  A word of warning: tested for region = 'br' (Brazil) with no date filters, it was found that 1106
  songs made it to the top 200 board at some moment (D2). From these, only 796 were linked to their
  corresponding record in D1. The remaining 310 unlinked tracks were manually inspected: it seems
  that these tracks indeed have no corresponding record in D1. This result suggests that there is a
  problem in the processes being used to collect the datasets, at least for the region of interest.

"""

import os
import sys
import numpy as np
import sharedDefs as ud

from os         import listdir, makedirs, remove
from os.path    import join, isfile, isdir, exists
from random     import seed
from datetime   import datetime

from sharedDefs import ECO_SEED
from sharedDefs import setupEssayConfig, getEssayParameter, setEssayParameter, overrideEssayParameter
from sharedDefs import getMountedOn, serialise, saveAsText, stimestamp, tsprint, saveLog, dict2text
from sharedDefs import loadAudioFeatures, loadDailyRankings, mapURL2ID

def main():

  # determines the simulation parameters
  param_sourcepath     = [getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'datasets', 'spotify']
  param_targetpath     = [getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']
  param_feature_file   = 'tracks.csv'
  param_feature_fields = ['acousticness',     'danceability', 'duration_ms', 'energy',   'tempo',
                          'instrumentalness', 'release_date', 'liveness',    'loudness', 'mode',
                          'speechiness',      'explicit',     'popularity',  'valence',  'key']
  param_topn_file      = 'data.csv'
  param_topn_fields    = ['Region', 'Date', 'URL', 'Track Name', 'Artist', 'Position', 'Streams']
  param_topn_region    = ['br']
  param_topn_from      = '1922-01-01'
  param_topn_to        = '2021-12-31'

  # ensures the folder where results will be saved is available and empty
  if(exists(join(*param_targetpath))):
    for f in listdir(join(*param_targetpath)):
      remove(join(*param_targetpath, f))
  else:
    makedirs(join(*param_targetpath))

  # loads Spotify's Audio Features dataset (D1)
  ud.LogBuffer = []
  tsprint("Loading Spotify's Audio Track Features dataset")
  (features, id2name, name2id) = loadAudioFeatures(param_sourcepath, param_feature_file, param_feature_fields)
  tsprint('-- audio features of {0} songs have been loaded.'.format(len(features)))

  # loads the Spotify's Worldwide Daily Song Ranking dataset (D2)
  # -- only records associated to the target region and period are considered
  tsprint("Loading Spotify's Daily Song Rankings dataset")
  tsprint('-- considering streams made by users from {0} to {1} in {2}'.format(param_topn_from,
                                                                               param_topn_to,
                                                                               param_topn_region))
  (rankings, timeline, songs) = loadDailyRankings(param_sourcepath,
                                                  param_topn_file,
                                                  param_topn_region,
                                                  param_topn_from,
                                                  param_topn_to)

  tsprint('-- {0} daily rankings have been loaded.'.format(len(rankings)))
  tsprint('-- {0} highly popular tracks have been identified.'.format(len(songs)))

  # builds the relationship between the datasets
  tsprint('Linking songs reported in daily rankings to their feature vectors')
  (url2id, failures, cases, samples) = mapURL2ID(songs, id2name, name2id)
  tsprint('-- {0} popular songs have been identified.'.format(len(songs)))
  tsprint('-- {0} popular songs were linked to their feature vector'.format(len(url2id)))
  tsprint('-- {0} popular songs remain unlinked'.format(len(failures)))
  tsprint('-- {0}'.format(failures), verbose=False)

  # saves the data
  tsprint('Saving results')

  # -- data from D1
  serialise(features, join(*param_targetpath, 'features'))
  serialise(id2name,  join(*param_targetpath, 'id2name'))
  serialise(name2id,  join(*param_targetpath, 'name2id'))

  # -- data from D2
  serialise(rankings, join(*param_targetpath, 'rankings'))
  serialise(timeline, join(*param_targetpath, 'timeline'))
  serialise(songs,    join(*param_targetpath, 'songs'))

  # -- data on the relationship between D1 and D2
  serialise(url2id,   join(*param_targetpath, 'url2id'))
  serialise(failures, join(*param_targetpath, 'failures'))

  # -- data for manual inspection of quality
  saveAsText(dict2text(cases),   join(*param_targetpath, 'cases.csv'))
  saveAsText(dict2text(samples), join(*param_targetpath, 'samples.csv'))

  tsprint('Job completed.')
  saveLog(join(*param_targetpath, 'preprocess.log'))

if __name__ == "__main__":

  main()
