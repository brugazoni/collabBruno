"""
  `estimate.py`

  syntax .: python estimate.py <config file>
  example : python estimate.py ../configs/general_T01_C0.cfg

  In `P01`/`simulate.py', the simulated user study is predicated on a "surprise threshold",
  which is an idealised construct that corresponds to the degree of surprise above which:
  (1) longer explanations are systematically preferred by participants, or
  (2) participants systematically explore a larger share of additional information (motivation)

  This script aims to estimate that threshold using datasets collected from the Spotify platform.
  The process is itself predicated on a number of conventions, premises and assumptions, which
  we make explicit in a logbook entry.

"""
import os
import sys
import numpy as np
import sharedDefs as ud

from os         import listdir, makedirs, remove
from os.path    import join, isfile, isdir, exists
from random     import seed, sample
from datetime   import datetime

from sharedDefs import ECO_SEED
from sharedDefs import setupEssayConfig, getEssayParameter, setEssayParameter, overrideEssayParameter
from sharedDefs import getMountedOn, deserialise, serialise, saveLog, saveAsText
from sharedDefs import tsprint, stimestamp, dict2text, headerfy
from sharedDefs import buildDataset, in_hull, estimateHullDistribs, plotHull

from scipy.spatial   import ConvexHull
from sklearn.cluster import KMeans

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
  param_sourcepath     = getEssayParameter('PARAM_TARGETPATH') # reads/writes in the same folder
  param_targetpath     = getEssayParameter('PARAM_TARGETPATH')
  param_feature_fields = getEssayParameter('PARAM_FEATURE_FIELDS')
  param_ignorelinks    = getEssayParameter('PARAM_IGNORELINKS')
  param_minpopularity  = getEssayParameter('PARAM_MINPOPULARITY')
  param_samplingprobs  = getEssayParameter('PARAM_SAMPLINGPROBS')
  param_dims           = getEssayParameter('PARAM_DIMS')
  param_epss           = getEssayParameter('PARAM_EPSS')

  # overrides parameters recovered from the config file with environment variables
  param_minpopularity  = overrideEssayParameter('PARAM_MINPOPULARITY')
  param_samplingprobs  = overrideEssayParameter('PARAM_SAMPLINGPROBS')
  param_dims           = overrideEssayParameter('PARAM_DIMS')
  param_epss           = overrideEssayParameter('PARAM_EPSS')
  
  # ensures the journal slot (where all executions are recorded) is available
  essay_beginning_ts = stimestamp()
  slot  = join('..', 'journal', essayid, configid, essay_beginning_ts)
  if(not exists(slot)): makedirs(slot)

  # adjusts the output directory to account for essay and config IDs
  param_targetpath += [essayid, configid]

  # initialises the random number generator
  seed(ECO_SEED)

  #---------------------------------------------------------------------------------------------
  # This is where the job is actually done; the rest is boilerpate
  #---------------------------------------------------------------------------------------------

  # loads preprocessed data
  tsprint('Loading preprocessed data')
  features = deserialise(join(*param_sourcepath, 'features'))
  url2id   = deserialise(join(*param_sourcepath, 'url2id'))

  # remove links that failed manual inspection
  for urlID in param_ignorelinks:
    url2id.pop(urlID)

  for (ndims, epsilon) in zip(param_dims, param_epss):

    # builds the dataset that will be employed by the estimation processes
    tsprint('Building the dataset')
    (P, Q, samples, ev) = buildDataset(url2id, features, param_feature_fields, ndims, epsilon,
                                       param_samplingprobs, param_minpopularity)
    (allPopIDs, allRegIDs, popIDs, regIDs) = samples
    tsprint('-- {0:5d} out of {1:6d} popular items included in the sample P'.format(len(popIDs), len(allPopIDs)))
    tsprint('-- {0:5d} out of {1:6d} regular items included in the sample Q'.format(len(regIDs), len(allRegIDs)))

    # determines which points in Q are interior/exterior to the hull
    tsprint('Determining which points are interior (or exterior) to the hull induced from P')
    (interior, summary, hull) = in_hull(Q, P)
    tsprint('-- {0:5d} items in sample Q are interior to the hull induced from P'.format(summary['interior']))
    tsprint('-- {0:5d} items in sample Q are exterior to the hull induced from P'.format(summary['exterior']))

    # estimates the distribution of max-distances from items interior/exterior to the hull
    #                                               to items representing hull vertices
    tsprint('Estimating surprise distributions for popular and regular items')
    stats, rawData = estimateHullDistribs(hull, Q, interior, popIDs, regIDs, features, param_feature_fields)

    # presents the results
    tsprint('Plotting the results')
    stats['explained_variance'] = sum(ev)
    plotHull(hull, Q, interior, stats, rawData, join(*param_targetpath, 'panel_{0}d'.format(ndims)))

    # ensures the folder where inspection data will be saved is available and empty
    newFolder = join(*param_targetpath, '{0}D'.format(ndims))
    if(exists(newFolder)):
      for f in listdir(newFolder):
        remove(join(newFolder, f))
    else:
      makedirs(newFolder)

    # saves the results
    serialise(P,        join(newFolder, 'P'))
    serialise(Q,        join(newFolder, 'Q'))
    serialise(samples,  join(newFolder, 'samples'))
    serialise(interior, join(newFolder, 'interior'))
    serialise(hull,     join(newFolder, 'hull'))
    serialise(stats,    join(newFolder, 'stats'))

  #---------------------------------------------------------------------------------------------
  # That's all, folks! The job has been done, we are closing for the day.
  #---------------------------------------------------------------------------------------------

  tsprint('Job completed.')
  saveLog(join(*param_targetpath, 'estimate.log'))

if __name__ == "__main__":

  main(sys.argv[1])
