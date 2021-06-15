"""
  `preproces.py`

  syntax .: python preproces.py <config file>
  example : python preproces.py ../configs/general_T01_C0.cfg

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
from sharedDefs import loadAudioFeatures, loadDailyRankings, mapURL2ID, buildReverso

def main(configFile):

  ud.LogBuffer = []

  # parses the config file
  tsprint('Running essay with specs recovered from [{0}]\n'.format(configFile))
  if(not isfile(configFile)):
    print('Command line parameter is not a file: {0}'.format(configFile))
    exit(1)
  tsprint('Processing essay configuration file [{0}]\n{1}'.format(configFile, setupEssayConfig(configFile)))

  # recovers attributes that identify the essay
  essayid  = getEssayParameter('ESSAY_ESSAYID')
  configid = getEssayParameter('ESSAY_CONFIGID')
  scenario = getEssayParameter('ESSAY_SCENARIO')
  replicas = getEssayParameter('ESSAY_RUNS')

  # recovers parameters related to the problem instance
  param_sourcepath     = getEssayParameter('PARAM_SOURCEPATH')
  param_targetpath     = getEssayParameter('PARAM_TARGETPATH')
  param_feature_file   = getEssayParameter('PARAM_FEATURE_FILE')
  param_feature_fields = getEssayParameter('PARAM_FEATURE_FIELDS')
  param_topn_file      = getEssayParameter('PARAM_TOPN_FILE')
  param_topn_fields    = getEssayParameter('PARAM_TOPN_FIELDS')
  param_topn_regions   = getEssayParameter('PARAM_TOPN_REGIONS')
  param_topn_from      = getEssayParameter('PARAM_TOPN_FROM')
  param_topn_to        = getEssayParameter('PARAM_TOPN_TO')
  param_vsm_common     = getEssayParameter('PARAM_VSM_COMMON')
  param_vsm_pairs      = getEssayParameter('PARAM_VSM_PAIRS')
  param_vsm_stopwords  = getEssayParameter('PARAM_VSM_STOPWORDS')

  # ensures the journal slot (where all executions are recorded) is available
  essay_beginning_ts = stimestamp()
  slot  = join('..', 'journal', essayid, configid, essay_beginning_ts)
  if(not exists(slot)): makedirs(slot)

  # adjusts the output directory to account for essay and config IDs
  param_targetpath += [essayid, configid]

  # ensures the folder where results will be saved is available and empty
  if(exists(join(*param_targetpath))):
    for f in listdir(join(*param_targetpath)):
      remove(join(*param_targetpath, f))
  else:
    makedirs(join(*param_targetpath))

  # initialises the random number generator
  seed(ECO_SEED)

  #---------------------------------------------------------------------------------------------
  # This is where the job is actually done; the rest is boilerpate
  #---------------------------------------------------------------------------------------------

  # loads Spotify's Audio Features dataset (D1)
  tsprint("Loading Spotify's Audio Track Features dataset")
  (features, id2name, name2id) = loadAudioFeatures(param_sourcepath, param_feature_file, param_feature_fields)
  vsmparams = (param_vsm_common, param_vsm_pairs, param_vsm_stopwords)
  reverso = buildReverso(id2name, vsmparams)
  tsprint('-- audio features of {0} songs have been loaded.'.format(len(features)))

  # loads the Spotify's Worldwide Daily Song Ranking dataset (D2)
  # -- only records associated to the target region and period are considered
  tsprint("Loading Spotify's Daily Song Rankings dataset")
  tsprint('-- considering streams made by users from {0} to {1} in {2}'.format(param_topn_from,
                                                                               param_topn_to,
                                                                               param_topn_regions))
  (rankings, timeline, songs) = loadDailyRankings(param_sourcepath,
                                                  param_topn_file,
                                                  param_topn_regions,
                                                  param_topn_from,
                                                  param_topn_to)

  tsprint('-- {0} daily rankings have been loaded.'.format(len(rankings)))
  tsprint('-- {0} highly popular tracks have been identified.'.format(len(songs)))

  # builds the relationship between the datasets
  tsprint('Linking songs reported in daily rankings to their feature vectors')
  (url2id, failures, cases, samples) = mapURL2ID(songs, id2name, name2id, vsmparams)
  tsprint('-- {0} popular songs have been identified.'.format(len(songs)))
  tsprint('-- {0} popular songs were linked to their feature vector'.format(len(url2id)))
  tsprint('-- {0} popular songs remain unlinked'.format(len(failures)))
  tsprint('-- {0}'.format(failures), verbose=False)

  # saves the data
  tsprint('Saving results')

  # -- processed data from D1
  serialise(features, join(*param_targetpath, 'features'))
  serialise(id2name,  join(*param_targetpath, 'id2name'))
  serialise(name2id,  join(*param_targetpath, 'name2id'))
  serialise(reverso,  join(*param_targetpath, 'reverso'))

  # -- processed data from D2
  serialise(rankings, join(*param_targetpath, 'rankings'))
  serialise(timeline, join(*param_targetpath, 'timeline'))
  serialise(songs,    join(*param_targetpath, 'songs'))

  # -- data on the relationship between D1 and D2
  serialise(url2id,   join(*param_targetpath, 'url2id'))
  serialise(failures, join(*param_targetpath, 'failures'))

  # -- data for manual inspection of quality
  saveAsText(dict2text(cases),   join(*param_targetpath, 'cases.csv'))
  saveAsText(dict2text(samples), join(*param_targetpath, 'samples.csv'))

  #---------------------------------------------------------------------------------------------
  # That's all, folks! The job has been done, we are closing for the day.
  #---------------------------------------------------------------------------------------------

  tsprint('Job completed.')
  saveLog(join(*param_targetpath, 'preprocess.log'))

if __name__ == "__main__":

  main(sys.argv[1])
