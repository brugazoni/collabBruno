import sys
import webbrowser
import numpy as np

from os.path     import join, isfile, isdir, exists
from sharedDefs  import tsprint, setupEssayConfig, getEssayParameter, overrideEssayParameter
from sharedDefs  import deserialise
from sharedDefs  import playHull

ECO_SEED       = 23

def main(ndims, configFile):

  # locks the random number generator
  np.random.seed(ECO_SEED)

  # loads app parameters
  tsprint('Retrieving parameters')
  setupEssayConfig(configFile)
  essayid              = getEssayParameter('ESSAY_ESSAYID')
  configid             = getEssayParameter('ESSAY_CONFIGID')
  param_sourcepath     = getEssayParameter('PARAM_TARGETPATH')
  param_feature_fields = getEssayParameter('PARAM_FEATURE_FIELDS')
  param_saveit         = overrideEssayParameter('PARAM_SAVEIT')
  param_sourcepath    += [essayid, configid, '{0}D'.format(ndims)]
  param_targetpath     = param_sourcepath

  # loads required data
  tsprint('Loading preprocessed data')
  Q        = deserialise(join(*param_sourcepath, 'Q'))
  hull     = deserialise(join(*param_sourcepath, 'hull'))
  stats    = deserialise(join(*param_sourcepath, 'stats'))
  interior = deserialise(join(*param_sourcepath, 'interior'))

  # creating the animation
  tsprint('Creating an animation of the {0}D projection'.format(ndims))
  playHull(hull, Q, interior, stats, join(*param_targetpath, 'itemspace_{0}d.mp4'.format(ndims)), param_saveit)

if(__name__ == '__main__'):

  ndims = int(sys.argv[1])
  configFile = sys.argv[2]

  if(ndims < 3):
    tsprint('-- animation can only be created for data in 3+ dimensions')
  else:
    main(ndims, configFile)
