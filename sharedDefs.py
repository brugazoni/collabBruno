import os
import pickle
import codecs
import numpy as np
import scipy as sp

from copy         import copy
from random       import seed, random
from pandas       import read_csv
from datetime     import datetime, timedelta
from itertools    import chain
from collections  import OrderedDict, defaultdict
from configparser import RawConfigParser
from urllib.parse import urlparse

ECO_SEED = 23
ECO_PRECISION = 1E-9
ECO_DATETIME_FMT = '%Y%m%d%H%M%S' # used in logging
ECO_RAWDATEFMT   = '%Y-%m-%d'     # used in file/memory operations
ECO_FIELDSEP     = ','


#-----------------------------------------------------------------------------------------------------------
# General purpose definitions - I/O helper functions
#-----------------------------------------------------------------------------------------------------------

LogBuffer = [] # buffer where all tsprint messages are stored

def stimestamp():
  return(datetime.now().strftime(ECO_DATETIME_FMT))

def stimediff(finishTs, startTs):
  return str(datetime.strptime(finishTs, ECO_DATETIME_FMT) - datetime.strptime(startTs, ECO_DATETIME_FMT))

def tsprint(msg, verbose=True):
  buffer = '[{0}] {1}'.format(stimestamp(), msg)
  if(verbose):
    print(buffer)
  LogBuffer.append(buffer)

def resetLog():
  LogBuffer = []

def saveLog(filename):
  saveAsText('\n'.join(LogBuffer), filename)

def serialise(obj, name):
  f = open(name + '.pkl', 'wb')
  p = pickle.Pickler(f)
  p.fast = True
  p.dump(obj)
  f.close()
  p.clear_memo()

def deserialise(name):
  f = open(name + '.pkl', 'rb')
  p = pickle.Unpickler(f)
  obj = p.load()
  f.close()
  return obj

def file2List(filename, separator = ',', erase = '"', _encoding = 'utf-8'):

  contents = []
  f = codecs.open(filename, 'r', encoding=_encoding)
  if(len(erase) > 0):
    for buffer in f:
      contents.append(buffer.replace(erase, '').strip().split(separator))
  else:
    for buffer in f:
      contents.append(buffer.strip().split(separator))
  f.close()

  return(contents)

def dict2text(d, header = ['Key', 'Value'], mask = '{0}\t{1}', parser = lambda e: [e]):

  content = [mask.format(*header)]

  for key in sorted(d):
    content.append(mask.format(key, *parser(d[key])))

  return '\n'.join(content)

def saveAsText(content, filename, _encoding='utf-8'):
  f = codecs.open(filename, 'w', encoding=_encoding)
  f.write(content)
  f.close()

def getMountedOn():

  if('PARAM_MOUNTEDON' in os.environ):
    res = os.environ['PARAM_MOUNTEDON'] + os.sep
  else:
    res = os.getcwd().split(os.sep)[-0] + os.sep

  return res

#-------------------------------------------------------------------------------------------------------------------------------------------
# General purpose definitions - interface to handle parameter files
#-------------------------------------------------------------------------------------------------------------------------------------------

# Essay Parameters hashtable
EssayParameters = {}

def setupEssayConfig(configFile = ''):

  # defines default values for some configuration parameters
  setEssayParameter('ESSAY_ESSAYID',  'None')
  setEssayParameter('ESSAY_CONFIGID', 'None')
  setEssayParameter('ESSAY_SCENARIO', 'None')
  setEssayParameter('ESSAY_RUNS',     '1')

  # overrides default values with user-defined configuration
  loadEssayConfig(configFile)

  return listEssayConfig()

def setEssayParameter(param, value):
  """
  Purpose: sets the value of a specific parameter
  Arguments:
  - param: string that identifies the parameter
  - value: its new value
    Premises:
    1) When using inside python code, declare value as string, independently of its true type.
       Example: 'True', '0.32', 'Rastrigin, normalised'
    2) When using parameters in Config files, declare value as if it was a string, but without the enclosing ''.
       Example: True, 0.32, Rastrigin, only Reproduction
  Returns: None
  """

  so_param = param.upper()

  # boolean-valued parameters
  if(so_param in ['PARAM_MASK_ERRORS']):

    so_value = eval(value[0]) if isinstance(value, list) else bool(value)

  # integer-valued parameters
  elif(so_param in ['ESSAY_RUNS', 'PARAM_MAXCORES', 'PARAM_MA_WINDOW']):

    so_value = eval(value[0])

  # floating-point-valued parameters
  elif(so_param in ['PARAM_new']):

    so_value = float(eval(value[0]))

  # parameters that requires eval expansion
  elif(so_param in ['PARAM_SOURCEPATH', 'PARAM_TARGETPATH', 'PARAM_TERRITORY', 'PARAM_POPSIZES',
                    'PARAM_OUTCOMES', 'PARAM_DATAFIELDS']):

    so_value = value

  # parameters that represent text
  else:

    so_value = value[0]

  EssayParameters[so_param] = so_value

def getEssayParameter(param):
  return EssayParameters[param.upper()]

def overrideEssayParameter(param):

  if(param in os.environ):
    param_value = os.environ[param]
    tsprint('-- option {0} replaced from {1} to {2} (environment variable setting)'.format(param,
                                                                                           getEssayParameter(param),
                                                                                           param_value))
    setEssayParameter(param, [str(param_value)])

  return getEssayParameter(param)

class OrderedMultisetDict(OrderedDict):

  def __setitem__(self, key, value):

    try:
      item = self.__getitem__(key)
    except KeyError:
      super(OrderedMultisetDict, self).__setitem__(key, value)
      return

    if isinstance(value, list):
      item.extend(value)
    else:
      item.append(value)

    super(OrderedMultisetDict, self).__setitem__(key, item)

def loadEssayConfig(configFile):

  """
  Purpose: loads essay configuration coded in a essay parameters file
  Arguments:
  - configFile: name and path of the configuration file
  Returns: None, but EssayParameters dictionary is updated
  """

  if(len(configFile) > 0):

    if(os.path.exists(configFile)):

      # initialises the config parser and set a custom dictionary in order to allow multiple entries
      # of a same key (example: several instances of GA_ESSAY_ALLELE
      config = RawConfigParser(dict_type = OrderedMultisetDict)
      config.read(configFile)

      # loads parameters codified in the ESSAY section
      for param in config.options('ESSAY'):
        setEssayParameter(param, config.get('ESSAY', param))

      # loads parameters codified in the PROBLEM section
      for param in config.options('PROBLEM'):
        setEssayParameter(param, config.get('PROBLEM', param))

      # expands parameter values that requires evaluation
      # parameters that may occur once, and hold lists or tuples
      if('PARAM_SOURCEPATH' in EssayParameters):
        EssayParameters['PARAM_SOURCEPATH']  = eval(EssayParameters['PARAM_SOURCEPATH'][0])

      if('PARAM_TARGETPATH' in EssayParameters):
        EssayParameters['PARAM_TARGETPATH']  = eval(EssayParameters['PARAM_TARGETPATH'][0])

      if('PARAM_DATAFIELDS' in EssayParameters):
        EssayParameters['PARAM_DATAFIELDS']  = eval(EssayParameters['PARAM_DATAFIELDS'][0])

      if('PARAM_TERRITORY' in EssayParameters):
        EssayParameters['PARAM_TERRITORY']  = eval(EssayParameters['PARAM_TERRITORY'][0])

      if('PARAM_POPSIZES' in EssayParameters):
        EssayParameters['PARAM_POPSIZES']  = eval(EssayParameters['PARAM_POPSIZES'][0])

      if('PARAM_OUTCOMES' in EssayParameters):
        EssayParameters['PARAM_OUTCOMES']  = eval(EssayParameters['PARAM_OUTCOMES'][0])

      # checks if configuration is ok
      (check, errors) = checkEssayConfig(configFile)
      if(not check):
        print(errors)
        exit(1)

    else:

      print('*** Warning: Configuration file [{1}] was not found'.format(configFile))

def checkEssayConfig(configFile):

  check = True
  errors = []
  errorMsg = ""

  # insert criteria below
  if(EssayParameters['ESSAY_ESSAYID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    param_name = 'ESSAY_ESSAYID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'].lower() not in configFile.lower()):
    check = False
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the config filename'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['PARAM_MA_WINDOW'] < 1):
    check = False
    param_name = 'PARAM_MA_WINDOW'
    restriction = 'be larger than zero'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  opts = ['Peddireddy', 'IB-forward']
  if(EssayParameters['PARAM_CORE_MODEL'] not in opts):
    check = False
    param_name = 'PARAM_CORE_MODEL'
    restriction = 'be one of {0}'.format(opts)
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  # summarises errors found
  if(len(errors) > 0):
    separator = "=============================================================================================================================\n"
    errorMsg = separator
    for i in range(0, len(errors)):
      errorMsg = errorMsg + errors[i]
    errorMsg = errorMsg + separator

  return(check, errorMsg)

# recovers the current essay configuration
def listEssayConfig():

  res = ''
  for e in sorted(EssayParameters.items()):
    res = res + "{0} : {1} (as {2})\n".format(e[0], e[1], type(e[1]))

  return res

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem specific definitions - preprocessing surveillance data
#-------------------------------------------------------------------------------------------------------------------------------------------

def loadAudioFeatures(sourcepath, featureFile, featurefields):

  # defines the field-to-type mapping
  def _cast(val, field):
    if(field == 'release_date'):
      res = float(val[0:4])
    elif(field in ['id', 'name']):
      res = str(val)
    elif(field in ['artists']):
      res = eval(val)
    else:
      res = float(val)
    return res

  # recovers the content of the features datafile
  df = read_csv(os.path.join(*sourcepath, featureFile))

  # parses the content to build the the dictionaries
  # features[itemID]  -> feature vector, and
  # id2name[itemID]   -> (name, artists)
  # name2id[name]     -> [(artists, itemID), ...]
  itemIDs  = []
  id2name  = {}
  features = {}
  name2id  = defaultdict(list)
  for _, row in df.iterrows():
    itemID  = _cast(row['id'],      'id')
    name    = _cast(row['name'],    'name')
    artists = _cast(row['artists'], 'artists')
    id2name[itemID]  = (name, artists)
    features[itemID] = [_cast(row[field], field) for field in featurefields]
    name2id[name].append((artists, itemID))
    #xxx name2id[name.lower()].append((artists, itemID))
    itemIDs.append(itemID)

  name2id = dict(name2id)

  # checks uniqueness of id (sorry, I found a lot of funny mismatches in the dataset)
  if(len(itemIDs) != len(set(itemIDs))):
    raise ValueError

  # applies standardisation to the vector representation
  M  = np.array([features[itemID] for itemID in itemIDs])
  mu = np.mean(M, 0)
  sd = np.std(M, 0, ddof=1)
  M  = (M - mu) / sd

  # feeding the results back to the dataset
  for i in range(len(itemIDs)):
    itemID = itemIDs[i]
    features[itemID] = M[i]

  return (features, id2name, name2id)

def loadDailyRankings(sourcepath, rankingFile, regions, date_from, date_to):

  # defines the field-to-type mapping
  def _cast(val, field):
    if(field == 'Date'):
      res = datetime.strptime(val, ECO_RAWDATEFMT)
    elif(field == 'URL'):
      res = urlparse(val).path.split('/')[-1]
    elif(field == 'Position'):
      res = int(val)
    else:
      res = str(val)
    return res

  # defines the period of time that will be considered
  date_lb = _cast(date_from, 'Date')
  date_ub = _cast(date_to,   'Date')

  # recovers the content of the features datafile
  df = read_csv(os.path.join(*sourcepath, rankingFile))

  # parses the content to build the the dictionaries
  # rankings[date] -> [(position, urlID), ...], ordered by increasing position
  # songs[urlID] -> (songName, songArtist)
  rankings = defaultdict(list)
  songs    = defaultdict(list)
  for _, row in df.iterrows():
    region     = _cast(row['Region'], 'Region')
    date       = _cast(row['Date'], 'Date')
    songName   = _cast(row['Track Name'], 'Track Name')
    songArtist = _cast(row['Artist'], 'Artist')
    if(region in regions and date >= date_lb and date <= date_ub):
      urlID    = _cast(row['URL'], 'URL')
      position = _cast(row['Position'], 'Position') #xxx use streams instead? works as weight
      rankings[date].append((position, urlID))
      songs[urlID].append((songName, songArtist))

  # creates the timeline and sorts the rankings (from most to least popular)
  timeline = sorted(rankings)
  for date in timeline:
    rankings[date].sort(key = lambda e: e[0]) #xxx position or streams?
  rankings = dict(rankings)

  # ensures url terminal serves as a key
  for urlID in songs:
    if(len(set(songs[urlID])) == 1):
      songs[urlID] = songs[urlID][0]
    else:
      print(urlID, songs[urlID])
      raise ValueError
  songs = dict(songs)

  return (rankings, timeline, songs)

def getIDExact(songName, name2id, params = None):
  return name2id[songName]

def getIDLower(songName, name2id, params = None):
  _name2Name = params
  return name2id[_name2Name[songName.lower()]]

def getIDRelaxed(songName, name2id, params = None):
  (threshold, pairs, stopWords, names) = params
  L = []
  s_ = set(substitute(songName.lower(), pairs).split()).difference(stopWords)
  for name in names:
    _s  = names[name]
    ss  = s_.intersection(_s)
    try:
      val = sim(ss, s_) + sim(ss, _s)
    except ZeroDivisionError:
      val = 0.0
    if(val >= threshold):
      L.append((val, name2id[name]))
  L.sort(key = lambda e: -e[0])
  res = list(chain(*[itemID for (_, itemID) in L]))
  return res

def sim(ss, other):
  return len(ss)/len(other)

def hasArtistExact(songArtist, artists, params = None):
  return (songArtist in artists)

def hasArtistRelaxed(songArtist, artists, params = None):
  L = [1 if songArtist.lower() in e.lower() or
                     e.lower() in songArtist.lower()
         else 0 for e in artists]

  return (sum(L) > 0)

def substitute(s, pairs):
  for (substr, to) in pairs:
    s = s.replace(substr, to)
  return s

def mapURL2ID(songs, id2name, name2id):

  # defines the basic procedure to link urlID and itemID
  # -- both url2id and failures are updated
  url2id = {}

  def _do(scope, songs, name2id, id2name, url2id, getID, hasArtist, params = None, verbose = False):
    failures = []
    for urlID in scope:
      (songName, songArtist) = songs[urlID]
      try:
        found = False
        for (artists, itemID) in getID(songName, name2id, params):
          if(hasArtist(songArtist, artists, params)):
            url2id[urlID] = itemID
            found = True
            if(verbose):
              tsprint(report(urlID, songs, url2id, id2name))
            break
      except KeyError:
        None

      if(not found):
        failures.append(urlID)

    return failures

  # links urlID to itemID when there is perfect match using song names
  last = 0
  tsprint('-- first pass')
  scope  = list(songs)
  failures = _do(scope, songs, name2id, id2name, url2id, getIDExact, hasArtistExact)
  tsprint('   contributed {0} links'.format(len(url2id) - last))
  last = len(url2id)

  # links urlID to itemID when there is exact name match and relaxed artist match
  tsprint('-- second pass')
  scope = failures
  failures = _do(scope, songs, name2id, id2name, url2id, getIDExact, hasArtistRelaxed)
  tsprint('   contributed {0} links'.format(len(url2id) - last))
  last = len(url2id)

  # links urlID to itemID by matching song names in lower case
  tsprint('-- third pass')
  params = {name.lower(): name for name in name2id}
  scope = failures
  failures = _do(scope, songs, name2id, id2name, url2id, getIDLower, hasArtistExact, params)
  tsprint('   contributed {0} links'.format(len(url2id) - last))
  last = len(url2id)

  # links urlID to itemID using relaxed matching for both name and artist (keep at the end, very slow)
  tsprint('-- fourth pass')
  pairs = [('ao vivo', ''), ('participação especial', ''), ('(', ''), (')', ''), ('[', ''), (']', ''), (' - ', ''), ('original motion picture', ''), ('official song', ''), ('radio edit', ''), ('soundtrack', ''), ('...', ''), (';', '')]
  stopWords = ['remix', 'remaster', 'remastered', 'acústica', 'acústico', 'acoustic', 'feat.', 'album', 'version', 'edit', 'editada', 'participação']
  params = (1.0, pairs, stopWords, {name: set(substitute(name.lower(), pairs).split()).difference(stopWords) for name in name2id})
  scope = failures
  failures = _do(scope, songs, name2id, id2name, url2id, getIDRelaxed, hasArtistRelaxed, params, verbose = True)
  tsprint('')
  tsprint('   contributed {0} links'.format(len(url2id) - last))
  last = len(url2id)

  # runs a sanity check on the obtained results (as said elsewhere, I am not trusting the dataset)
  ECO_MATCH_CLASS_0 = '0. urlID not linked to an itemID'
  ECO_MATCH_CLASS_1 = '1. exact name match, single artist'
  ECO_MATCH_CLASS_2 = '2. exact name match, first artist match'
  ECO_MATCH_CLASS_3 = '3. exact name match, artist in the list'
  ECO_MATCH_CLASS_4 = '4. lower name match, single artist'
  ECO_MATCH_CLASS_5 = '5. lower name match, first artist match'
  ECO_MATCH_CLASS_6 = '6. lower name match, artist in the list'
  ECO_MATCH_CLASS_7 = '7. exact name match, relaxed artist match'
  ECO_MATCH_CLASS_8 = '8. relaxed name match, artist in the list'
  ECO_MATCH_CLASS_9 = '9. relaxed name match, relaxed artist match'
  ECO_MATCH_CLASS_U = 'U. unclassified'

  cases   = defaultdict(int)
  samples = defaultdict(list)

  for urlID in songs:
    (songName, songArtist) = songs[urlID]
    try:
      itemID = url2id[urlID]
    except KeyError:
      itemID = None
      cases[ECO_MATCH_CLASS_0] += 1
      samples[ECO_MATCH_CLASS_0].append(urlID)

    if(itemID is not None):

      (name, artists) = id2name[itemID]
      if(songName == name and songArtist == artists[0] and len(artists) == 1):
        cases[ECO_MATCH_CLASS_1]  += 1

      elif(songName == name and songArtist == artists[0] and len(artists) > 1):
        cases[ECO_MATCH_CLASS_2]  += 1

      elif(songName == name and songArtist in artists and len(artists) > 1):
        cases[ECO_MATCH_CLASS_3]  += 1

      elif(songName.lower() == name.lower() and songArtist == artists[0] and len(artists) == 1):
        cases[ECO_MATCH_CLASS_4]  += 1

      elif(songName.lower() == name.lower() and songArtist == artists[0] and len(artists) > 1):
        cases[ECO_MATCH_CLASS_5]  += 1

      elif(songName.lower() == name.lower() and songArtist in artists and len(artists) > 1):
        cases[ECO_MATCH_CLASS_6]  += 1

      elif(songName == name and hasArtistRelaxed(songArtist, artists)):
        cases[ECO_MATCH_CLASS_7]  += 1
        samples[ECO_MATCH_CLASS_7].append(urlID)

      elif((songName.lower() in name.lower() or name.lower() in songName.lower()) and hasArtistExact(songArtist, artists)):
        cases[ECO_MATCH_CLASS_8]  += 1
        samples[ECO_MATCH_CLASS_8].append(urlID)

      elif((songName.lower() in name.lower() or name.lower() in songName.lower()) and hasArtistRelaxed(songArtist, artists)):
        cases[ECO_MATCH_CLASS_9]  += 1
        samples[ECO_MATCH_CLASS_9].append(urlID)

      else:
        cases[ECO_MATCH_CLASS_U] += 1
        samples[ECO_MATCH_CLASS_U].append(urlID)

  return (url2id, failures, cases, samples)

def report(urlID, songs, url2id, id2name):

  (songName, songArtist, itemID, name, artists) = 5 * ['-']

  try:
    (songName, songArtist) = songs[urlID]
    itemID = url2id[urlID]
    (name, artists) = id2name[itemID]

  except:
    None

  content = []
  content.append('')
  content.append('-- [rankings] song name ..: {0}'.format(songName))
  content.append('-- [features] song name ..: {0}'.format(name))
  content.append('-- [rankings] url ID .....: {0}'.format(urlID))
  content.append('-- [features] item ID ....: {0}'.format(itemID))
  content.append('-- [rankings] song artist : {0}'.format(songArtist))
  content.append('-- [features] song artist : {0}'.format(artists))

  return '\n'.join(content)

