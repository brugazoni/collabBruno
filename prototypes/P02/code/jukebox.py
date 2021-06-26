import sys
import webbrowser
import numpy as np
import sharedDefs as ud

from copy        import copy
from os.path     import join, isfile, isdir, exists
from collections import namedtuple, defaultdict
from sharedDefs  import setupEssayConfig, getEssayParameter
from sharedDefs  import deserialise, serialise
from sharedDefs  import substitute, buildReverso
from sklearn.decomposition import PCA

ECO_SEED       = 23
ECO_CURSOR     = ':'
ECO_PERPAGE    = 30
ECO_PERROW     = 125
ECO_TOPN       = 5
#ECO_LARGESS    = 172230 # 586672
ECO_GUESTUSER  = 'guest'
ECO_WILDCARD   = '*'

ACTION_NONE    = ''
ACTION_SEARCH  = 's'
ACTION_FILTER  = 'f'
ACTION_END     = 'q'
ACTION_LISTW   = 'lw'
ACTION_LISTP   = 'lp'
ACTION_LISTU   = 'lu'
ACTION_P2W     = 'p2w'
ACTION_W2P     = 'w2p'
ACTION_ADDTOP  = 'a'
ACTION_DELFRP  = 'd'
ACTION_SUGGEST = 'r'
ACTION_TOGGLE  = 't'
ACTION_VECTOR  = 'v'
ACTION_PLAY    = 'p'
ACTION_COMPARE = 'c'
ACTION_USER    = 'u'
ACTION_CHNDIMS = 'z'

Database  = namedtuple('Database',  ['profiles', 'reverso', 'id2name', 'url2id', 'features', 'featureFields'])

def euclidist(v, w):
  return np.linalg.norm(v-w)

def cosdist(v, w):
  sim = v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w))
  return 0.5 * (1 - sim)

class Jukebox:

  def __init__(self, distfn, dataset, threshold, pairs, ss):

    self.distfn    = distfn
    self.dataset   = dataset
    self.threshold = threshold
    self.pairs     = pairs
    self.ss        = ss

    self.working   = []
    self.filters   = defaultdict(list)
    self.current   = ECO_GUESTUSER
    self.itemvecs  = self.dataset.features
    self.itemsize  = len(self.dataset.featureFields)
    self.maxdims   = self.itemsize

    self.dataset.profiles[self.current] = []
    self.profile   = self.dataset.profiles[self.current]
    self.popular   = list(set(dataset.url2id.values()))

  def _averagevec(self):
    if(len(self.profile) > 0):
      res = np.mean([self.itemvecs[itemID] for itemID in self.profile], 0)
    else:
      res = np.zeros(self.itemsize)
    return res

  def _recommend(self, tokens):

    # recovers the position of the 'popularity' field in the feature vector
    _idx = self.dataset.featureFields.index('popularity')

    # samples the catalog for suggestions
    L = []
    v = self._averagevec()
    #candidates  = np.random.choice(list(self.dataset.id2name), self.ss).tolist()
    #candidates += self.popular
    candidates  = list(self.dataset.id2name)
    candidates  = [itemID for itemID in candidates if itemID not in self.profile]
    for itemID in set(candidates):
      w = self.itemvecs[itemID]
      L.append((itemID, self.distfn(v,w), self.dataset.features[itemID][_idx]))

    L.sort(key = lambda e: e[1])

    # adds the top ECO_TOPN "highly relevant, but unsurprising" suggestions
    suggestions = []

    if('r' in tokens):
      for i in range(ECO_TOPN):
        suggestions.append(L[i][0])

    # adds the top ECO_TOPN "highly surprising yet hopefully relevant" suggestions
    if('s' in tokens):
      last = ECO_TOPN
      for i in range(ECO_TOPN):
        for k in range(last, self.ss):
          if(L[k][2] <= self.threshold):
            suggestions.append(L[k][0])
            last = k+1
            break

    return suggestions

  def _parse(self, cmd):
    tokens = list(filter(None, [token for token in cmd.strip().lower().split(' ')]))
    try:
      action = tokens[0]
      params = tokens[1:]
    except IndexError:
      action = ACTION_NONE
      params = []

    if(action in ['rr', 'rs', 'fa', 'fy', 'fp', 'f.']):
      params = [action[1]] + params
      action = action[0]

    return (action, params)

  def _listItems(self, itemIDs):
    for i in range(len(itemIDs)):
      itemID = itemIDs[i]
      self._displayItem(i, itemID)
    return True

  def _listItemDetails(self, itemIDs):
    for i in range(len(itemIDs)):
      itemID = itemIDs[i]
      self._displayItem(i, itemID)
      print('  {0}'.format(self.itemvecs[itemID]))
    return True

  def _displayItem(self, i, itemID):
    (name, artists, year) = self.dataset.id2name[itemID]
    buffer = '  {0:2d}: {1} {2} {3:40} {4}'.format(i, itemID, year, name, artists)
    print(buffer[:ECO_PERROW])
    return None

  def execute(self, cmd):

    # parses the command line
    res = True
    (action, params) = self._parse(cmd)

    try:
      if(  action == ACTION_SEARCH):  res = self.action_search(params)
      elif(action == ACTION_FILTER):  res = self.action_setFilter(params)
      elif(action == ACTION_LISTW):   res = self.action_listWorking()
      elif(action == ACTION_LISTP):   res = self.action_listProfile()
      elif(action == ACTION_LISTU):   res = self.action_listUsers()
      elif(action == ACTION_P2W):     res = self.action_profile2working()
      elif(action == ACTION_W2P):     res = self.action_working2profile()
      elif(action == ACTION_ADDTOP):  res = self.action_addToProfile(params)
      elif(action == ACTION_DELFRP):  res = self.action_delFrProfile(params)
      elif(action == ACTION_VECTOR):  res = self.action_showVector(params)
      elif(action == ACTION_SUGGEST): res = self.action_suggest(params)
      elif(action == ACTION_COMPARE): res = self.action_compare(params)
      elif(action == ACTION_TOGGLE):  res = self.action_toggle()
      elif(action == ACTION_USER):    res = self.action_switchUser(params)
      elif(action == ACTION_PLAY):    res = self.action_play(params)
      elif(action == ACTION_CHNDIMS): res = self.action_changeItemspace(params)
      elif(action == ACTION_NONE):    res = True
      elif(action == ACTION_END):     res = False
      else:
        print('--- command not recognised.')

    except:
      print('-- error processing command.')
      None

    return res

  def action_search(self, params):

    # a token can be:
    # -- part of the song title,
    # -- part of the artist name, or
    # -- the year of release
    # -- a wildcard
    tokens = params

    # checks if the wildcard has been employed
    if(tokens[0] == ECO_WILDCARD):
      candidates = {itemID: 1 for itemID in self.dataset.id2name}
    else:
      # seeks tracks that contains the tokens in their names
      candidates = defaultdict(int)
      for token in tokens:
        try:
          for itemID in self.dataset.reverso[token]:
            candidates[itemID] += 1
        except KeyError:
          None
    if(len(candidates) > 0):
      maxMatches = max(candidates.values())
    else:
      maxMatches = 0

    # removes candidate items that do not match the current artist-filter
    numOfTokens = len(self.filters['a'])
    if(numOfTokens > 0):
      # applies the artist-filter as an AND filter
      newCandidates = {}
      for itemID in candidates:
        (_, artists, _) = self.dataset.id2name[itemID]
        terms = substitute(' '.join(artists), self.pairs).lower().split(' ')
        matches = sum([1 if token in terms else 0 for token in self.filters['a']])
        if(matches == numOfTokens):
          newCandidates[itemID] = maxMatches + 1

      candidates = newCandidates

    # removes candidate items that do not match the current year-filter
    numOfTokens = len(self.filters['y'])
    if(numOfTokens > 0):
      # applies the year-filter as an OR filter
      newCandidates = {}
      for itemID in candidates:
        (_, _, year) = self.dataset.id2name[itemID]
        if(year in self.filters['y']):
          newCandidates[itemID] = maxMatches + 1

      candidates = newCandidates

    L = sorted([(itemID, candidates[itemID]) for itemID in candidates], key = lambda e: -e[1])
    self.working = [itemID for (itemID, _) in L][0:ECO_PERPAGE]

    return self.action_listWorking()

  def action_setFilter(self, params):

    if(len(params) == 0):
      # clears current filters
      self.filters = defaultdict(list)

    elif(params[0] in ['a', 'y']):
      # adds new tokens to the filter
      self.filters[params[0]] += params[1:]

    elif(params[0] in ['.', 'p']):
      # prints current filters
      for e in self.filters:
        print('  {0}: {1}'.format(e, self.filters[e]))

    else:
      print('-- invalid parameters')

    return True

  def action_listWorking(self):
    return self._listItems(self.working)

  def action_listProfile(self):
    return self._listItems(self.profile)

  def action_listUsers(self):

    print('-- active user is: {0}'.format(self.current))
    for userID in self.dataset.profiles:
      profileSize = len(self.dataset.profiles[userID])
      buffer = '  {0:15}: {1:2d} item(s)'.format(userID, profileSize)
      print(buffer[:ECO_PERROW])

    return True

  def action_profile2working(self):
    self.working = copy(self.profile)
    return True

  def action_working2profile(self):
    self.dataset.profiles[self.current] = copy(self.working)
    self.profile = self.dataset.profiles[self.current]
    return True

  def action_showVector(self, params):
    # shows feature vectors of items in the profile
    # if params has no tokens, the average profile vector is shown
    if(len(params) == 0)  :
      print('   {0}'.format(self._averagevec()))
    else:
      for token in params:
        i = int(token)
        print()
        self._listItems([self.profile[i]])
        print('   {0}'.format(self.itemvecs[self.profile[i]]))
    return True

  def action_addToProfile(self, tokens):
    # a token is the index of an item in the working area
    # that must be added to the profile
    for token in tokens:
      i = int(token)
      try:
        self.profile.append(self.working[i])
      except KeyError:
        print('-- {0} not a proper index of the buffering area'.format(token))
        None
    return True

  def action_delFrProfile(self, tokens):
    # a token is the index of an item in the profile that must be removed
    L = []
    idxs = [int(token) for token in tokens]
    for i in range(len(self.profile)):
      if(i not in idxs):
        L.append(self.profile[i])
    self.dataset.profiles[self.current] = L
    self.profile = self.dataset.profiles[self.current]
    return True

  def action_suggest(self, params):
    if(len(params) == 0):
      params = ['r', 's']
    self.working = self._recommend(params)
    return self.action_listWorking()

  def action_toggle(self):
    if(self.distfn == euclidist):
      self.distfn = cosdist
      print('--- distance function toggled to cosine distance')
    else:
      self.distfn = euclidist
      print('--- distance function toggled to euclidean distance')
    return True

  def action_compare(self, params):
    if(len(params) == 0):
      print('--- parameter missing')
      return True
    if(len(params) >  1):
      print('--- command accepts a single parameter; remaining ones being ignored')

    i = int(params[0])
    itemID = self.working[i]
    w = self.itemvecs[itemID]
    self._listItems([itemID])
    print('  {0}'.format(w))
    print()

    L = []
    for itemID in self.profile:
      v = self.itemvecs[itemID]
      L.append((itemID, self.distfn(v,w)))
    L.sort(key = lambda e: e[1])
    itemIDs = [itemID for (itemID, _) in L]
    return self._listItemDetails(itemIDs)

  def action_switchUser(self, params):
    if(len(params) == 0):
      print('--- parameter missing')
      return True
    if(len(params) >  1):
      print('--- command accepts a single parameter; remaining ones being ignored')

    # makes the new user the active user
    userID = params[0]
    if(userID not in self.dataset.profiles or userID == ECO_GUESTUSER):
      self.dataset.profiles[userID] = []
    self.profile = self.dataset.profiles[userID]
    self.current = userID
    return True

  def action_play(self, params):
    if(len(params) == 0):
      print('--- parameter missing')
      return True
    if(len(params) >  1):
      print('--- command accepts a single parameter; remaining ones being ignored')
    i = int(params[0])
    itemID = self.working[i]
    url = 'https://open.spotify.com/track/{0}'.format(itemID)
    webbrowser.get('chrome').open(url)
    return True

  def action_changeItemspace(self, params):
    if(len(params) == 0):
      print('--- parameter missing')
      return True
    if(len(params) >  1):
      print('--- command accepts a single parameter; remaining ones being ignored')

    ndims = int(params[0])
    if(ndims in [0, self.maxdims]):
      self.itemvecs  = self.dataset.features
      print('--- changed to original dimensionality')
    elif(ndims > 1 and ndims < self.maxdims):
      itemIDs = sorted(self.dataset.features)
      Q   = np.vstack([self.dataset.features[itemID] for itemID in itemIDs])
      pca = PCA(n_components = ndims, svd_solver = 'arpack', random_state = ECO_SEED)
      Q_  = pca.fit_transform(Q)
      self.itemvecs = {itemIDs[i]: Q_[i] for i in range(len(itemIDs))}
      self.itemsize = ndims
      print('--- changed to {0}-dimensional itemspace'.format(ndims))
    else:
      print('--- invalid parameter')

    return True

def main(configFile):

  print('-- Welcome to Spotify Jukebox!')

  # locks the random number generator
  np.random.seed(ECO_SEED)

  # loads app parameters
  setupEssayConfig(configFile)
  essayid  = getEssayParameter('ESSAY_ESSAYID')
  configid = getEssayParameter('ESSAY_CONFIGID')
  param_sourcepath     = getEssayParameter('PARAM_TARGETPATH')
  param_sourcepath    += [essayid, configid]
  param_feature_fields = getEssayParameter('PARAM_FEATURE_FIELDS')
  param_vsm_common     = getEssayParameter('PARAM_VSM_COMMON')
  param_vsm_pairs      = getEssayParameter('PARAM_VSM_PAIRS')
  param_vsm_stopwords  = getEssayParameter('PARAM_VSM_STOPWORDS')
  param_largess        = getEssayParameter('PARAM_LARGESS')
  param_threshold      = getEssayParameter('PARAM_THRESHOLD')
  param_browser        = getEssayParameter('PARAM_BROWSER')

  # loads required data
  print('   Loading preprocessed data')
  features = deserialise(join(*param_sourcepath, 'features'))
  id2name  = deserialise(join(*param_sourcepath, 'id2name'))
  url2id   = deserialise(join(*param_sourcepath, 'url2id'))

  try:
    reverso  = deserialise(join(*param_sourcepath, 'reverso'))
  except FileNotFoundError:
    vsmparams = (param_vsm_common, param_vsm_pairs, param_vsm_stopwords)
    reverso = buildReverso(id2name, vsmparams)
    serialise(reverso, join(*param_sourcepath, 'reverso'))

  try:
    profiles = deserialise(join(*param_sourcepath, 'jukebox'))
  except FileNotFoundError:
    profiles = {ECO_GUESTUSER: []}

  # initialises the background resources
  webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(join(*param_browser)))
  dataset  = Database(profiles, reverso, id2name, url2id, features, param_feature_fields)
  jukebox = Jukebox(euclidist, dataset, param_threshold, param_vsm_pairs, param_largess)

  # attends to the command line
  print('   Ready. Enter your command and press enter.')
  online = True
  while online:
    print(ECO_CURSOR, end=' ')
    try:
      usrcmd = input()
      #online = run(usrcmd, resources, dataset)
      online = jukebox.execute(usrcmd)
    except EOFError:
      online = False

  # saves the profiles built during the session
  serialise(dataset.profiles, join(*param_sourcepath, 'jukebox'))

if(__name__ == '__main__'):

  main(sys.argv[1])
