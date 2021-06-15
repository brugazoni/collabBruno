import re
import os
import pickle
import codecs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from copy           import copy
from scipy          import stats
from random         import random, sample
from pandas         import read_csv
from datetime       import datetime, timedelta
from itertools      import chain
from matplotlib     import animation
from collections    import OrderedDict, defaultdict, namedtuple
from configparser   import RawConfigParser
from urllib.parse   import urlparse
from scipy.spatial  import ConvexHull, Delaunay, convex_hull_plot_2d
from scipy.optimize import linprog

from sklearn.decomposition import PCA

ECO_SEED = 23
ECO_PRECISION = 1E-9
ECO_DATETIME_FMT = '%Y%m%d%H%M%S' # used in logging
ECO_RAWDATEFMT   = '%Y-%m-%d'     # used in file/memory operations
ECO_FIELDSEP     = ','

FakeHull = namedtuple('FakeHull', ['points', 'vertices', 'simplices'])

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

def headerfy(mask):
  res = re.sub('\:\d+\.\d+f', '', mask)
  res = re.sub('\:\d+d', '', res)
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
  if(so_param in ['PARAM_SAVEIT']):

    so_value = eval(value[0]) if isinstance(value, list) else bool(value)

  # integer-valued parameters
  elif(so_param in ['ESSAY_RUNS', 'PARAM_DIM_ITEMSPACE']):

    so_value = eval(value[0])

  # floating-point-valued parameters
  elif(so_param in ['PARAM_MINPOPULARITY', 'PARAM_THRESHOLD', 'PARAM_VSM_COMMON']):

    so_value = float(eval(value[0]))

  # parameters that requires eval expansion
  elif(so_param in ['PARAM_SOURCEPATH',    'PARAM_TARGETPATH',   'PARAM_FEATURE_FIELDS',
                    'PARAM_TOPN_FIELDS',   'PARAM_TOPN_REGIONS', 'PARAM_IGNORELINKS',
                    'PARAM_SAMPLINGPROBS', 'PARAM_DIMS',         'PARAM_EPSS',
                    'PARAM_BROWSER',       'PARAM_VSM_PAIRS',    'PARAM_VSM_STOPWORDS']):




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

      if('PARAM_FEATURE_FIELDS' in EssayParameters):
        EssayParameters['PARAM_FEATURE_FIELDS']  = eval(EssayParameters['PARAM_FEATURE_FIELDS'][0])

      if('PARAM_TOPN_FIELDS' in EssayParameters):
        EssayParameters['PARAM_TOPN_FIELDS']  = eval(EssayParameters['PARAM_TOPN_FIELDS'][0])

      if('PARAM_TOPN_REGIONS' in EssayParameters):
        EssayParameters['PARAM_TOPN_REGIONS']  = eval(EssayParameters['PARAM_TOPN_REGIONS'][0])

      if('PARAM_IGNORELINKS' in EssayParameters):
        EssayParameters['PARAM_IGNORELINKS']  = eval(EssayParameters['PARAM_IGNORELINKS'][0])

      if('PARAM_SAMPLINGPROBS' in EssayParameters):
        EssayParameters['PARAM_SAMPLINGPROBS']  = eval(EssayParameters['PARAM_SAMPLINGPROBS'][0])

      if('PARAM_DIMS' in EssayParameters):
        EssayParameters['PARAM_DIMS']  = eval(EssayParameters['PARAM_DIMS'][0])

      if('PARAM_EPSS' in EssayParameters):
        EssayParameters['PARAM_EPSS']  = eval(EssayParameters['PARAM_EPSS'][0])

      if('PARAM_BROWSER' in EssayParameters):
        EssayParameters['PARAM_BROWSER']  = eval(EssayParameters['PARAM_BROWSER'][0])

      if('PARAM_VSM_PAIRS' in EssayParameters):
        EssayParameters['PARAM_VSM_PAIRS']  = eval(EssayParameters['PARAM_VSM_PAIRS'][0])

      if('PARAM_VSM_STOPWORDS' in EssayParameters):
        EssayParameters['PARAM_VSM_STOPWORDS']  = eval(EssayParameters['PARAM_VSM_STOPWORDS'][0])

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
# Problem specific definitions - preprocessing Spotify datasets
#-------------------------------------------------------------------------------------------------------------------------------------------

def loadAudioFeatures(sourcepath, featureFile, featurefields):

  # source record. encoding: 18. means "field 18, included in the preprocessed data"
  #                          19- means "field 19, ignored by the process"

  # 0.  1.    2.          3.           4.        5.       6           7.            8.
  # id, name, popularity, duration_ms, explicit, artists, id_artists, release_date, danceability,

  # 9.      10.  11.       12.   13.          14.           15.               16.       17.
  # energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence,

  # 18.    19-
  # tempo, time_signature

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

  # parses the content to build these dictionaries:
  # features[itemID]  -> feature vector, and
  # id2name[itemID]   -> (name, artists, year)
  # name2id[name]     -> [(itemID, artists, year), ...]

  id2name  = {}
  features = {}
  name2id  = defaultdict(list)
  itemIDs  = defaultdict(int)
  for _, row in df.iterrows():
    itemID  = _cast(row['id'],      'id')
    name    = _cast(row['name'],    'name')
    artists = _cast(row['artists'], 'artists')
    year    =       row['release_date'][0:4]
    id2name[itemID]  = (name, artists, year)
    name2id[name].append((itemID, artists, year))
    features[itemID] = [_cast(row[field], field) for field in featurefields]
    itemIDs[itemID] += 1

  # checks uniqueness of id (sorry, I found a lot of quality issues in the dataset)
  duplicates = [itemID for itemID in itemIDs if itemIDs[itemID] > 1]
  if(len(duplicates) > 0):
    tsprint('   {0}'.format(duplicates))
    tsprint('-- {0} duplicated keys found'.format(len(duplicates)))
    raise ValueError
  name2id = dict(name2id)
  itemIDs = list(itemIDs)

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

def buildReverso(id2name, vsmparams):

  (_, pairs, stopWords) = vsmparams
  reverso = defaultdict(list)
  for itemID in id2name:

    (name, artists, year) = id2name[itemID]

    tokens = set(substitute(name.lower(), pairs).split()).difference(stopWords)
    for token in tokens:
      reverso[token].append(itemID)

    for artist in artists:
      tokens = set(substitute(artist.lower(), pairs).split()).difference(stopWords)
      for token in tokens:
        reverso[token].append(itemID)

    reverso[year].append(itemID)

  reverso = dict(reverso)

  return reverso

def loadDailyRankings(sourcepath, rankingFile, regions, date_from, date_to):

  # source record. encoding: 1. means "field 1, included in the preprocessed data"
  #                          1- means "field 1, ignored by the process"

  # 1.        2.          3.      4.       5.   6.    7.
  # Position, Track Name, Artist, Streams, URL, Date, Region

  # defines the field-to-type mapping
  def _cast(val, field):
    if(field == 'Date'):
      res = datetime.strptime(val, ECO_RAWDATEFMT)
    elif(field == 'URL'):
      res = urlparse(val).path.split('/')[-1]
    elif(field in ['Position', 'Streams']):
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
  # rankings[date] -> [(urlID, position, streams, region), ...], ordered by increasing position
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
      position = _cast(row['Position'], 'Position')
      streams  = _cast(row['Streams'],  'Streams')
      rankings[date].append((urlID, position, streams, region))
      songs[urlID].append((songName, songArtist))

  # creates the timeline and sorts the rankings (from most to least popular)
  timeline = sorted(rankings)
  for date in timeline:
    rankings[date].sort(key = lambda e: e[1])
  rankings = dict(rankings)

  # ensures url terminal serves as a key
  for urlID in songs:
    if(len(set(songs[urlID])) == 1):
      songs[urlID] = songs[urlID][0]
    else:
      # a fatal error is raised in case the uniqueness of the urlID ~ (song,artist) is violated
      print(urlID, songs[urlID])
      raise ValueError
  songs = dict(songs)

  return (rankings, timeline, songs)

def getIDExact(songName, name2id, params = None):
  return name2id[songName]

def getIDLower(songName, name2id, name2Name = None):
  return name2id[name2Name[songName.lower()]]

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

def mapURL2ID(songs, id2name, name2id, vsmparams):

  # defines the basic procedure to link urlID and itemID
  # -- both url2id and failures are updated
  def _do(scope, songs, name2id, id2name, url2id, getID, hasArtist, params = None, verbose = False):
    failures = []
    for urlID in scope:
      (songName, songArtist) = songs[urlID]
      try:
        found = False
        for (itemID, artists, year) in getID(songName, name2id, params):
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

  url2id = {}

  # links urlID to itemID when there is a perfect match between song names
  last = 0
  tsprint('-- first pass: matching songs by name and artist, perfect string match')
  scope  = list(songs)
  failures = _do(scope, songs, name2id, id2name, url2id, getIDExact, hasArtistExact)
  tsprint('   contributed {0} links'.format(len(url2id) - last))
  last = len(url2id)

  # links urlID to itemID when there is exact name match and relaxed artist match
  tsprint('-- second pass: matching songs by exact name match and relaxed artist match')
  scope = failures
  failures = _do(scope, songs, name2id, id2name, url2id, getIDExact, hasArtistRelaxed)
  tsprint('   contributed {0} links'.format(len(url2id) - last))
  last = len(url2id)

  # links urlID to itemID by matching song names in lower case
  tsprint('-- third pass: matching songs by relaxed name match and exact artist match')
  params = {name.lower(): name for name in name2id}
  scope = failures
  failures = _do(scope, songs, name2id, id2name, url2id, getIDLower, hasArtistExact, params)
  tsprint('   contributed {0} links'.format(len(url2id) - last))
  last = len(url2id)

  # links urlID to itemID using relaxed matching for both name and artist
  # (must be kept at the end because it is very expensive/slow)
  tsprint('-- fourth pass: matching songs by name and artist using VSM search with stop words')
  (commonality, pairs, stopWords) = vsmparams
  params = (commonality, pairs, stopWords,
            {name: set(substitute(name.lower(), pairs).split()).difference(stopWords) for name in name2id})
  scope = failures
  failures = _do(scope, songs, name2id, id2name, url2id, getIDRelaxed, hasArtistRelaxed, params, verbose = True)
  tsprint('')
  tsprint('   contributed {0} links'.format(len(url2id) - last))
  last = len(url2id)

  # links urlID to itemID by matching identifiers
  tsprint('-- fifth pass: matchinh songs by exact identifier match')
  matchedIDs = set(failures).intersection(list(id2name))
  for urlID in matchedIDs:
    url2id[urlID] = urlID
    failures.remove(urlID)
    tsprint(report(urlID, songs, url2id, id2name))
  tsprint('   contributed {0} links'.format(len(matchedIDs)))
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
  ECO_MATCH_CLASS_U = 'U. undetermined'

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

      (name, artists, year) = id2name[itemID]
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

  tsprint('Applying sanity check')
  for case in sorted(cases):
    tsprint('{0:3d} items classified as {1}'.format(cases[case], case))

  tsprint('Unlinked items (for manual inspection)')
  for urlID in failures:
    tsprint(report(urlID, songs, url2id, id2name))

  return (url2id, failures, cases, samples)

def report(urlID, songs, url2id, id2name):

  (songName, songArtist, itemID, name, artists) = 5 * ['-']

  try:
    (songName, songArtist) = songs[urlID]
    itemID = url2id[urlID]
    (name, artists, year) = id2name[itemID]

  except:
    None

  content = []
  content.append('')
  content.append('-- [rankings] song name ..: {0}'.format(songName))
  content.append('-- [features] song name ..: {0}'.format(name))
  content.append('-- [rankings] url  ID ....: {0}'.format(urlID))
  content.append('-- [features] item ID ....: {0}'.format(itemID))
  content.append('-- [rankings] song artist : {0}'.format(songArtist))
  content.append('-- [features] song artist : {0}'.format(artists))

  return '\n'.join(content)

#-----------------------------------------------------------------------------------------------------------
# General purpose definitions - convex hull in high dimensional spaces
#-----------------------------------------------------------------------------------------------------------

def distance(v, w):
  return np.linalg.norm(v - w)

def in_hull(Q, P):
  """
  Determines which points in Q are interior to the hull induced from P
  """

  nd = len(Q[0])

  if(nd < 10):
    # computes the convex hull of P
    tsprint('-- computing the convex hull around P')
    hull = ConvexHull(P)
    tsprint('-- item density interior to the hull induced from P is {0:8.5f}'.format(len(P)/hull.volume))
  else:
    hull = FakeHull(points = P, vertices = list(range(len(P))), simplices=[])

  W = np.array([hull.points[i] for i in hull.vertices])

  if(nd < 8):
    # employs the triagulation approach
    tsprint('-- applying triangulation to classify points in Q as interior or exterior')
    (interior, summary) = in_hull_tri(Q, W)
  else:
    # employs the linear programming approach
    tsprint('-- applying linear programming to classify points in Q as interior or exterior')
    (interior, summary) = in_hull_lp(Q, W)

  return (interior, summary, hull)

def in_hull_tri(Q, W):
  tri = Delaunay(W)
  interior = tri.find_simplex(Q) >= 0
  summary = {'interior': sum([1 for e in interior if e]), 'exterior': sum([1 for e in interior if not e])}
  return interior, summary

def in_hull_lp(Q, W):

    n_vertices = len(W)

    c = np.zeros(n_vertices)
    A = np.r_[W.T, np.ones((1, n_vertices))]

    interior = []
    summary = {'interior': 0, 'exterior': 0}
    for q in Q:
      b  = np.r_[q, np.ones(1)]
      try:
        lp = linprog(c, A_eq=A, b_eq=b)
        success = lp.success
      except ValueError:
        success = False

      interior.append(success)
      summary['interior' if success else 'exterior'] += 1

    return (interior, summary)

def estimateHullDistribs(hull, Q, interior, popIDs, regIDs, features, featureFields, samplingProb = 1.0):

  # produces data that are rendered by plotHull on 'Panel 2'
  W = np.array([hull.points[i] for i in hull.vertices])
  (sample_int, sample_ext) = ([], [])
  for (isInterior, v) in zip(interior, Q):
    if(np.random.rand() <= samplingProb):
      surpriseub = max([distance(v, w) for w in W])
      (sample_int if isInterior else sample_ext).append(surpriseub)

  ss_int = len(sample_int)
  ss_ext = len(sample_ext)
  ci_int = bs.bootstrap(np.array(sample_int), stat_func=bs_stats.mean)
  ci_ext = bs.bootstrap(np.array(sample_ext), stat_func=bs_stats.mean)
  tsprint('-- mu internal distribution: {0} (out of {1} samples)'.format(ci_int.value, ss_int))
  tsprint('               95% interval: [{0}, {1}]'.format(ci_int.lower_bound, ci_int.upper_bound))
  tsprint('-- mu external distribution: {0} (out of {1} samples)'.format(ci_ext.value, ss_ext))
  tsprint('               95% interval: [{0}, {1}]'.format(ci_ext.lower_bound, ci_ext.upper_bound))

  stats = {'int.sample': sample_int, 'int.ci': ci_int,
           'ext.sample': sample_ext, 'ext.ci': ci_ext}

  # produces data that are rendered by plotHull on 'Panel 3' and 'Panel 4'
  idx_popularity = featureFields.index('popularity')
  rawData = {'P': [], 'Q': []}

  for i in range(len(popIDs)):
    itemID = popIDs[i]
    popularity = features[itemID][idx_popularity]
    v = hull.points[i]
    position = True
    surpriseub = max([distance(v, w) for w in W])
    rawData['P'].append((itemID, position, popularity, surpriseub))

  for i in range(len(regIDs)):
    itemID = regIDs[i]
    popularity = features[itemID][idx_popularity]
    v = Q[i]
    position = interior[i]
    surpriseub = max([distance(v, w) for w in W])
    rawData['Q'].append((itemID, position, popularity, surpriseub))

  return stats, rawData

def plotHull(hull, Q_, interior, distStats, rawData, filename):

  # unpacks parameters
  (unitsizew, unitsizeh) = (1.940, 1.916)
  (nrows, ncols) = (6, 12)
  nd = Q_.shape[1]

  pop_pttrn_1      = 'c-'
  pop_pttrn_2      = 'c:'
  sample_pttrn_1   = 'm-'
  sample_pttrn_2   = 'm:'
  interior_pttrn_1 = 'b-'
  interior_pttrn_2 = 'b:'
  exterior_pttrn_1 = 'r-'
  exterior_pttrn_2 = 'r:'

  fig = plt.figure(figsize=(ncols * unitsizew, nrows * unitsizeh))
  innerseps = {'left': 0.06, 'bottom': 0.06, 'right': 0.94, 'top': 0.96, 'wspace': 0.90, 'hspace': 1.10}
  plt.subplots_adjust(left   = innerseps['left'],
                      bottom = innerseps['bottom'],
                      right  = innerseps['right'],
                      top    = innerseps['top'],
                      wspace = innerseps['wspace'],
                      hspace = innerseps['hspace'])

  gs = fig.add_gridspec(nrows, ncols)
  plt.xkcd()

  panel1 = fig.add_subplot(gs[0:3, 0:6])
  panel1.set_title('Item space (projection from {0}-dimensional data, ev = {1:5.3f})'.format(nd, distStats['explained_variance']))
  panel1.set_xlim(-5.0,  8.0)
  panel1.set_ylim(-5.0, 25.0)

  panel2 = fig.add_subplot(gs[3:, 0:6])
  panel2.set_title('Distribution of expected upper bound surprise')
  panel2.set_xlabel('Expected upper bound surprise')
  panel2.set_ylabel('Frequency')
  panel2.set_xlim(0.0, 12.0)
  panel2.set_ylim(0.0,  1.8)

  panel3 = fig.add_subplot(gs[0:3, 6:])
  panel3.set_title('Expected upper bound surprise conditioned on popularity')
  panel3.set_xlabel('Popularity')
  panel3.set_ylabel('Expected upper bound surprise')
  panel3.set_xlim(-1.6, 3.6)
  panel3.set_ylim(3.5, 10.0)

  panel4 = fig.add_subplot(gs[3:, 6:])
  panel4.set_title('Distribution of popularity')
  panel4.set_xlabel('Popularity')
  panel4.set_ylabel('Frequency')
  panel4.set_yscale('log')
  panel4.set_xlim(-1.6, 3.6)
  panel4.set_ylim(   1, 1E3)

  # ensures item vectores are projected to 2D
  if(nd == 2):
    Q = Q_
    V = hull.points
  else:
    pca = PCA(n_components=2, svd_solver = 'arpack', random_state = ECO_SEED)
    Q = pca.fit_transform(Q_)
    V = pca.transform(hull.points)

  #------------------------------------------------------------------
  # Panel 1 - Item space
  #------------------------------------------------------------------

  # plots the points in Q that are exterior to Hull(P)
  (firstInt, firstExt) = (True, True)
  for (isInterior, v) in zip(interior, Q):
    if(not isInterior):
      if(firstExt):
        panel1.plot(v[0], v[1], exterior_pttrn_1.replace('-', '+'), label = 'Regular (exterior)')
        firstExt = False
      else:
        panel1.plot(v[0], v[1], exterior_pttrn_1.replace('-', '+'))

  # plots the highly popular items
  panel1.plot(V[:,0], V[:,1], pop_pttrn_1.replace('-', 'o'), label='Highly popular')

  # plots the points in Q that are interior to Hull(P)
  (firstInt, firstExt) = (True, True)
  for (isInterior, v) in zip(interior, Q):
    if(isInterior):
      if(firstInt):
        panel1.plot(v[0], v[1], interior_pttrn_1.replace('-', '+'), label = 'Popular (interior)')
        firstInt = False
      else:
        panel1.plot(v[0], v[1], interior_pttrn_1.replace('-', '+'))

  # if original data in 2D, plots the boundaries of the hull
  if(nd == 2):
    for simplex in hull.simplices:
      panel1.plot(V[simplex, 0], V[simplex, 1], interior_pttrn_2)

  panel1.legend(loc = 'upper right')

  #------------------------------------------------------------------
  # Panel 2 - Distribution of expected upper bound surprise
  #------------------------------------------------------------------

  # instead of histograms, induces distributions based on gaussian kernels fitted to samples
  sample_int = distStats['int.sample']
  sample_ext = distStats['ext.sample']
  method = 'silverman'

  try:
    kde1 = stats.gaussian_kde(sample_int, method)
    kde1_pattern = interior_pttrn_1
  except np.linalg.LinAlgError:
    # if singular matrix, just black out; not a good solution, so ...
    kde1 = lambda e: [0 for _ in e]
    kde1_pattern = interior_pttrn_2

  try:
    kde2 = stats.gaussian_kde(sample_ext, method)
    kde2_pattern = exterior_pttrn_1
  except np.linalg.LinAlgError:
    kde2 = lambda e: [0 for _ in e]
    kde2_pattern = exterior_pttrn_2

  x_lb = min(sample_int + sample_ext)
  x_ub = max(sample_int + sample_ext)
  x_grades = 200
  x_eval = np.linspace(x_lb, x_ub, num=x_grades)
  y_int = kde1(x_eval)
  y_ext = kde2(x_eval)
  panel2.plot(x_eval, y_ext, kde2_pattern, label='Regular (exterior)')
  panel2.plot(x_eval, y_int, kde1_pattern, label='Popular (interior)')

  mu_int = distStats['int.ci'].value
  mu_ext = distStats['ext.ci'].value
  y_ubi = y_int[[i for i in range(x_grades) if x_eval[i] >= mu_int][0]]
  y_ube = y_ext[[i for i in range(x_grades) if x_eval[i] >= mu_ext][0]]

  panel2.fill_betweenx((0, y_ubi), distStats['int.ci'].lower_bound,
                                   distStats['int.ci'].upper_bound, alpha=.13, color='g')

  panel2.fill_betweenx((0, y_ube), distStats['ext.ci'].lower_bound,
                                   distStats['ext.ci'].upper_bound, alpha=.13, color='g')

  panel2.axvline(mu_int, 0.0, y_ubi, color=kde1_pattern[0], linestyle=':')
  panel2.axvline(mu_ext, 0.0, y_ube, color=kde2_pattern[0], linestyle=':')

  panel2.legend(loc = 'upper right')

  #------------------------------------------------------------------
  # Panel 3 - Expected upper bound surprise conditioned on popularity
  #------------------------------------------------------------------

  def _rawData2hist(rawData, onlyPosition = None, aggregation = np.mean):
    temp = defaultdict(list)
    for (itemID, position, popularity, surpriseub) in rawData:
      if(onlyPosition is None or onlyPosition == position):
        temp[popularity].append(surpriseub)
    hist = sorted([(popularity, aggregation(temp[popularity])) for popularity in temp], key = lambda e: e[0])
    (x_vals, y_vals) = zip(*hist)
    return (x_vals, y_vals)

  (x_vals, y_vals) = _rawData2hist(rawData['Q'], onlyPosition = False)
  panel3.plot(x_vals, y_vals, exterior_pttrn_2, label='Regular (exterior)')

  (x_vals, y_vals) = _rawData2hist(rawData['Q'], onlyPosition = True)
  panel3.plot(x_vals, y_vals, interior_pttrn_2, label='Popular (interior)')

  (x_vals, y_vals) = _rawData2hist(rawData['Q'])
  panel3.plot(x_vals, y_vals, sample_pttrn_1, label='Whole Sample')

  (x_vals, y_vals) = _rawData2hist(rawData['P'])
  panel3.plot(x_vals, y_vals, pop_pttrn_2, label='Highly popular')

  panel3.legend(loc = 'upper right')

  #------------------------------------------------------------------
  # Panel 4 - Distribution of popularity
  #------------------------------------------------------------------
  f = lambda e: len(e)

  (x_vals, y_vals) = _rawData2hist(rawData['Q'], onlyPosition = False, aggregation = f)
  panel4.plot(x_vals, y_vals, exterior_pttrn_2, label='Regular (exterior)')

  (x_vals, y_vals) = _rawData2hist(rawData['Q'], onlyPosition = True, aggregation = f)
  panel4.plot(x_vals, y_vals, interior_pttrn_2, label='Popular (interior)')

  (x_vals, y_vals) = _rawData2hist(rawData['Q'], aggregation = f)
  panel4.plot(x_vals, y_vals, sample_pttrn_1, label='Whole Sample')

  (x_vals, y_vals) = _rawData2hist(rawData['P'], aggregation = f)
  panel4.plot(x_vals, y_vals, pop_pttrn_2, label='Highly popular')

  panel4.legend(loc = 'upper right')


  plt.savefig(filename, bbox_inches = 'tight')
  plt.close(fig)

  return None

def playHull(hull, Q_, interior, distStats, filename = None, saveit = True):

  # unpacks parameters
  #(unitsizew, unitsizeh) = (1.940, 1.916)
  (unitsizew, unitsizeh) = (2.000, 1.916)
  (nrows, ncols) = (3, 3)
  nd = Q_.shape[1]

  pop_pttrn_1      = 'c-'
  pop_pttrn_2      = 'c:'
  sample_pttrn_1   = 'm-'
  sample_pttrn_2   = 'm:'
  interior_pttrn_1 = 'b-'
  interior_pttrn_2 = 'b:'
  exterior_pttrn_1 = 'r-'
  exterior_pttrn_2 = 'r:'

  tsprint('-- projecting data to 3D itemspace')
  #plt.xkcd()
  fig = plt.figure(figsize=(ncols * unitsizew, nrows * unitsizeh))
  panel1 = fig.add_subplot(projection='3d')
  panel1.set_title('Item space (projection from {0}D data, ev = {1:5.3f})'.format(nd, distStats['explained_variance']))
  panel1.set_xlim(-5.0,  8.0)
  panel1.set_ylim(-5.0, 25.0)

  # ensures item vectores are projected to 3D
  if(nd < 3):
    return None
  elif(nd == 3):
    Q = Q_
    V = hull.points
  else:
    pca = PCA(n_components=3, svd_solver = 'arpack', random_state = ECO_SEED)
    Q = pca.fit_transform(Q_)
    V = pca.transform(hull.points)

  #------------------------------------------------------------------
  # Panel 1 - Item space
  #------------------------------------------------------------------

  # plots the points in Q that are exterior to Hull(P)
  tsprint('-- rendering items in 3D itemspace')
  (firstInt, firstExt) = (True, True)
  for (isInterior, v) in zip(interior, Q):
    if(not isInterior):
      if(firstExt):
        panel1.plot(v[0], v[1], v[2], exterior_pttrn_1.replace('-', '+'), label = 'Regular (exterior)')
        firstExt = False
      else:
        panel1.plot(v[0], v[1], v[2], exterior_pttrn_1.replace('-', '+'))

  # plots the highly popular items
  panel1.plot(V[:,0], V[:,1], V[:,2], pop_pttrn_1.replace('-', 'o'), label='Highly popular')

  # plots the points in Q that are interior to Hull(P)
  (firstInt, firstExt) = (True, True)
  for (isInterior, v) in zip(interior, Q):
    if(isInterior):
      if(firstInt):
        panel1.plot(v[0], v[1], v[2], interior_pttrn_1.replace('-', '+'), label = 'Popular (interior)')
        firstInt = False
      else:
        panel1.plot(v[0], v[1], v[2], interior_pttrn_1.replace('-', '+'))

  # if original data in 2D, plots the boundaries of the hull
  if(nd == 3):
    for simplex in hull.simplices:
      panel1.plot(V[simplex, 0], V[simplex, 1], V[simplex, 2], interior_pttrn_2)

  if(saveit):

    def rotate_azim(i):
      if(i < 360):
        panel1.view_init(elev=10., azim=i)
      else:
        panel1.view_init(azim=10., elev=i)
      print('.', end = '')
      return fig,

    tsprint('-- rendering and saving the animation')
    anim = animation.FuncAnimation(fig, rotate_azim, init_func=None,
                                   frames=720, interval=200, blit=True)

    anim.save(filename, fps=5, extra_args=['-vcodec', 'libx264'])

  else:

    plt.show()

  return None

def buildDataset(url2id, features, featureFields, n_components = 2, epsilon = 1.00, samplingProbs = (1.0, 1.0), minPopularity = 5.00):

  # minPopularity :- default value is set to a very very very very high level (>= 5-sigma)
  # https://www.theguardian.com/science/life-and-physics/2014/sep/15/five-sigma-statistics-bayes-particle-physics

  # obtains the sizes of the P and Q partitions (of U, the universe set)
  # -- P is the set of items that are known to be highly popular
  # -- Q is the complement of P; thus, Q may contain both popular and regular items
  allPopIDs = sorted(set(url2id.values()))
  allRegIDs = sorted(set(features).difference(allPopIDs))

  # to smooth the hull induced by P, (1) removes items from P whose relative distance to the nearest
  # neighbour is larger than a given constant epsilon, and (2) removes items with popularity lower
  # than a minimal value (minPopularity)
  OM = defaultdict(list)
  for itemIDrow in allPopIDs:
    for itemIDcol in allPopIDs:
      if(itemIDrow != itemIDcol):
        v = features[itemIDrow]
        w = features[itemIDcol]
        OM[itemIDrow].append((itemIDcol, distance(v, w)))

  smallest = []
  for itemID in OM:
    OM[itemID] = sorted(OM[itemID], key = lambda e: e[1])
    smallest.append(OM[itemID][0][1])

  smallest.sort()
  delta = smallest[int((len(smallest) - 1) * epsilon)]

  _idx_popularity = featureFields.index('popularity')
  for itemID in OM:
    if(OM[itemID][0][1] > delta or features[itemID][_idx_popularity] < minPopularity):
      # moves the item from P to Q
      allPopIDs.remove(itemID)
      allRegIDs.append(itemID)

  # samples P and Q according to specifications
  (sp_P, sp_Q) = samplingProbs
  popIDs = sample(allPopIDs, int(len(allPopIDs) * sp_P))
  regIDs = sample(allRegIDs, int(len(allRegIDs) * sp_Q))

  # obtains a high-dimensional version of the dataset
  P_ = np.vstack([features[itemID] for itemID in popIDs]) # sample of highly popular items
  Q_ = np.vstack([features[itemID] for itemID in regIDs]) # sample of regular items (may be popular or not)

  # reduces the dimensionality of the dataset, if required
  nd = P_.shape[1]
  if(n_components == 0):
    n_components = nd

  if(nd == n_components):
    P  = P_
    Q  = Q_
    ev = [1.0]
    tsprint('-- items represented as {0}-dimensional vectors'.format(nd))
  elif(nd > n_components):
    pca = PCA(n_components = n_components, svd_solver = 'arpack', random_state = ECO_SEED)
    Q  = pca.fit_transform(Q_)
    P  = pca.transform(P_)
    ev = pca.explained_variance_ratio_
    tsprint('-- number of dimensions reduced from {0} to {1}'.format(nd, n_components))
    tsprint('-- explained variance is {0:5.3f} ({1})'.format(sum(ev), ev))
  else:
    # if nd < n_components (i.e., we want to project to a higher dimensional space),
    # then the caller made a mistake
    raise ValueError

  samples = (allPopIDs, allRegIDs, popIDs, regIDs)

  return (P, Q, samples, ev)
