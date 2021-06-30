import sys
import webbrowser
import numpy as np

from os.path     import join, isfile, isdir, exists
from sharedDefs  import ECO_SEED
from sharedDefs  import tsprint, setupEssayConfig, getEssayParameter, overrideEssayParameter
from sharedDefs  import deserialise, serialise, saveAsText, dict2text
from sharedDefs  import loadSurveyData, applyQualityCriteria, measurements2text, testHypotheses

def main(configFile):

  # locks the random number generator
  np.random.seed(ECO_SEED)

  # loads app parameters
  tsprint('Retrieving parameters')
  setupEssayConfig(configFile)
  essayid           = getEssayParameter('ESSAY_ESSAYID')
  configid          = getEssayParameter('ESSAY_CONFIGID')
  param_sourcepath  = getEssayParameter('PARAM_SOURCEPATH')
  param_sourcepath  = param_sourcepath[:-1] + ['collected']
  param_survey_file = getEssayParameter('PARAM_SURVEY_FILE')
  param_targetpath  = getEssayParameter('PARAM_TARGETPATH')
  param_targetpath += [essayid, configid]
  param_youtubeok  =  getEssayParameter('PARAM_YOUTUBEOK')

  # loads required data
  tsprint('Loading preprocessed data')
  features = deserialise(join(*param_targetpath, 'features'))
  id2name  = deserialise(join(*param_targetpath, 'id2name'))
  name2id  = deserialise(join(*param_targetpath, 'name2id'))
  reverso  = deserialise(join(*param_targetpath, 'reverso'))

  # loads the collected data
  tsprint('Loading survey data')
  (caseRecord, rawMeasurements) = loadSurveyData(param_sourcepath, param_survey_file)
  saveAsText(measurements2text(rawMeasurements), join(*param_targetpath, 'rawMeasurements.csv'))
  tsprint('-- {0:3d} cases were recovered.'.format(len(caseRecord)))

  # applies data quality criteria to identify data to be discarded
  (rejected, measurements, accepted, discarded) = applyQualityCriteria(caseRecord, rawMeasurements, param_youtubeok)
  saveAsText(dict2text(rejected), join(*param_targetpath, 'rejected.csv'))
  discardedCases = len(set([caseID for (caseID, _) in rejected]))
  tsprint('-- {0:3d} cases    out of {1:3d} were found to fail some quality criterion.'.format(discardedCases, len(caseRecord)))
  tsprint('-- {0:3d} replicas out of {1:3d} were discarded.'.format(discarded, accepted + discarded))
  saveAsText(measurements2text(measurements), join(*param_targetpath, 'measurements.csv'))

  # analyses the data (to assess the relevant hypotheses)
  tsprint('Assessing relevant hypotheses')
  assessments = testHypotheses(measurements)

  # saves the processed data
  tsprint('Saving processed data')
  serialise(caseRecord,      join(*param_targetpath, 'caseRecord'))
  serialise(rawMeasurements, join(*param_targetpath, 'rawMeasurements'))
  serialise(rejected,        join(*param_targetpath, 'rejected'))
  serialise(measurements,    join(*param_targetpath, 'measurements'))


if(__name__ == '__main__'):

  configFile = sys.argv[1]
  main(configFile)
