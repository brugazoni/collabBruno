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
  param_feature_fields = ['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                          'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
                          'popularity', 'valence', 'release_date']
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

  # initialises the random number generator
  seed(ECO_SEED)

  # loads the Spotify Dataset 1922-2021 dataset (~600k tracks)
  # -- premise [P1] The selected audio-features are Tversky-relevant for all users
  ud.LogBuffer = []
  tsprint('Loading Spotify Dataset of audio features')
  (features, id2name, name2id) = loadAudioFeatures(param_sourcepath, param_feature_file, param_feature_fields)
  tsprint('-- audio features of {0} songs have been loaded.'.format(len(features)))

  # loads the Spotify's Worldwide Daily Song Ranking (WDSR) dataset
  # -- only records associated to the target population (associated to particular region) are considered
  # -- URL terminal node is used as song identifier (WSDR.id)
  tsprint('Loading Spotify Dataset of daily song rankings')
  tsprint('-- considering streams made by users from {0} to {1} in {2}'.format(param_topn_from, param_topn_to,
                                                                               param_topn_region))
  (rankings, timeline, songs) = loadDailyRankings(param_sourcepath, param_topn_file, param_topn_region,
                                                                    param_topn_from, param_topn_to)
  tsprint('-- {0} daily rankings have been loaded.'.format(len(rankings)))
  tsprint('-- {0} popular songs have been identified.'.format(len(songs)))

  # builds the relationship between the datasets
  tsprint('Linking songs reported in daily rankings to their feature vectors')
  (url2id, failures, cases, samples) = mapURL2ID(songs, id2name, name2id)
  tsprint('-- {0} popular songs have been identified.'.format(len(songs)))
  tsprint('-- {0} popular songs were linked to their feature vector'.format(len(url2id)))
  tsprint('-- {0} popular songs remain unlinked'.format(len(failures)))
  tsprint('-- {0}'.format(failures), verbose=False)

  # determines the convex hull that encloses all popular items in the target population
  # -- premise [P2] Every user has at least one popular item in their profile
  #for date in timeline:

  # determines set of items that are not enclosed in the hull

  # determines the distribution of shortest distance between items outside the hull and items within the hull

  # saves the data
  tsprint('Saving results')
  serialise(features, join(*param_targetpath, 'features'))
  serialise(id2name,  join(*param_targetpath, 'id2name'))
  serialise(name2id,  join(*param_targetpath, 'name2id'))

  serialise(rankings, join(*param_targetpath, 'rankings'))
  serialise(timeline, join(*param_targetpath, 'timeline'))
  serialise(songs,    join(*param_targetpath, 'songs'))

  serialise(url2id,   join(*param_targetpath, 'url2id'))
  serialise(failures, join(*param_targetpath, 'failures'))

  saveAsText(dict2text(cases),   join(*param_targetpath, 'cases.csv'))
  saveAsText(dict2text(samples), join(*param_targetpath, 'samples.csv'))
  saveLog(join(*param_targetpath, 'config.log'))

if __name__ == "__main__":

  main()
