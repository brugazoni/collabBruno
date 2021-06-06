"""
  `simulate.py`
  Simulates a user study to clarify the relationship between surprise
  and preferred explanation (shorter or longer) in recommender systems
  by exploring a user model (Agent). It also performs a hypothesis testing
  to check if the obtained results support the existence of the relationship
  (in this very idealised setting)

  syntax .: python simulate.py <number of agents> <number of queries> [<mock>]

  example : python simulate.py 15 6 mock
"""

import re
import os
import sys
import pickle
import codecs
import numpy as np
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from scipy       import stats
from random      import seed, sample, choice, shuffle
from pandas      import read_csv
from datetime    import datetime
from collections import defaultdict

# parameters used in generating and handling the mocked dataset

ECO_SEED  = 23     # seed for the random number generators
ECO_MU    = 0.0    # average value of a feature
ECO_SD    = 1.0    # spread of values of a feature
ECO_SPLIT = 0.2    # the fraction of the dataset that is allocated to the test partition
                   # the remaining items are allocated to the test partition
ECO_PROFILEVAR = 2 # used to specify the size of the profile partition
                   # indirectly controls the overlap among agents' profiles:
                   # the larger the value, the smaller the overlap
                   # (with overlap being measured with the Jaccard distance for sets)

# parameters used to specify the simulation
ECO_PROFILESIZE  = 10    # number of item in the user profile
ECO_LARGESAMPLE  = 1000  # the size of a sample from which recommendations are to be selected

# constants used to identify types of explanations
ECO_SHORTEXP     = 'short'
ECO_LONGEXP      = 'long'

# constants used to name agents
ECO_HUMAN_NAMES  = ['Alice',   'Beatriz', 'Claudio', 'Demetrio',  'Eduardo',
                    'Fabio',   'Gustavo', 'Helena',  'Isabela',   'Jose',
                    'Kamila',  'Laura',   'Marcos',  'Nina',      'Orestes',
                    'Paulo',   'Quesia',  'Ricardo', 'Sheila',    'Tomas',
                    'Ulric',   'Valeria', 'Wallace', 'Xuxa',      'Yasmin',
                    'Zuleika', 'Ana',     'Bianca',  'Cristiano', 'Diego'
                   ]

#--------------------------------------------------------------------------------------------------
# General purpose definitions - I/O helpers
#--------------------------------------------------------------------------------------------------

ECO_DATETIME_FMT = '%Y%m%d%H%M%S' # used in logging

LogBuffer = [] # buffer where all tsprint messages are stored

def stimestamp():
  return(datetime.now().strftime(ECO_DATETIME_FMT))

def tsprint(msg, verbose=True):
  buffer = '[{0}] {1}'.format(stimestamp(), msg)
  if(verbose):
    print(buffer)
  LogBuffer.append(buffer)

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

#--------------------------------------------------------------------------------------------------
# Problem-related definitions - data retrieval and data handling
#--------------------------------------------------------------------------------------------------

def cosdist(v, w):
  # positive-shifted cosine distance (returns values in the real interval [0, 1])
  # assumes v and w are numpy arrays with similar number of elements
  return (1 - v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w))) / 2

def euclideandist(v, w):
  return np.linalg.norm(v - w)

def getDataset(datasetParams, numOfAgents):

  # unpacks dataset parameters
  (dataset, sourcepath, sourcefile, sourceFields) = datasetParams

  # default sizes of the Spotify's Track Features dataset ~ 600k tracks
  numOfItems    = 586672
  numOfFeatures = 15

  def _standardise(features):
    itemIDs = list(features)
    M  = np.vstack([features[itemID] for itemID in itemIDs])
    mu = np.mean(M, 0)
    sd = np.std(M, 0, ddof=1)
    M  = (M - mu) / sd
    return M, itemIDs

  def _report(M, featureFields, alpha = 3E-4):
    """
    reports summary statistics of the standardised data
    alpha has been calibrated to the numOfItems, numOfFeatures, ECO_MU = 0.0 and ECO_SD = 1.0, meaning
    that a simulation with a mocked dataset (in which features are generated from normal distributions)
    attains the required results (i.e., 'Do not reject' for all fields)
    """
    nd = M.shape[1]
    mu = np.mean(M, 0)
    sd = np.std(M, 0, ddof=1)
    nt = ['Reject' if stats.normaltest(M[:,i]).pvalue < alpha else 'Do not reject' for i in range(nd)]
    mask = '{0:20}\t{1:6.3f}\t{2:6.3f}\t{3}'
    tsprint(headerfy(mask).format('Field', 'mu\t\t', 'sd\t\t', 'H: from normal dist?'))
    for i in range(len(featureFields)):
      field = featureFields[i]
      tsprint(mask.format(field, mu[i], sd[i], nt[i]))

    return None

  if('TESTSIMUL' in os.environ):

    pad = [0 for _ in range(numOfFeatures - 2)]
    u = np.array([ 1, 0] + pad) # profile base vector
    v = np.array([ 0, 1] + pad) # test  part. base vector
    w = np.array([-1, 0] + pad) # train part. base vector

    sizeProfilep = int(numOfAgents * ECO_PROFILESIZE * ECO_PROFILEVAR)
    sizeTestp    = int(ECO_SPLIT * numOfItems)
    sizeTrainp   = numOfItems - sizeTestp - sizeProfilep

    L = (
        [(k + 1, (k + 1) * w) for k in range(sizeTrainp)] +
        [(k + 1, (k + 1) * v) for k in range(sizeTrainp, sizeTrainp + sizeTestp)] +
        [(k + 1, (k + 1) * u) for k in range(sizeTrainp + sizeTestp, sizeTrainp + sizeTestp + sizeProfilep)]
        )

    features = {k: vector for (k, vector) in L}
    sourceFields = ['field {0}'.format(i) for i in range(numOfFeatures)]

  else:

    if(dataset == 'mocked'):

      # generates a mocked dataset containing "numOfItems" items,
      # each described by "numOfFeatures" features
      # *** assumes the features can be approximately described by normal distributions
      features = {itemID: np.random.normal(ECO_MU, ECO_SD, numOfFeatures) for itemID in range(numOfItems)}
      sourceFields = ['field {0}'.format(i) for i in range(numOfFeatures)]

      # reports sumary statistics of the mocked dataset
      itemIDs = list(features)
      M  = np.vstack([features[itemID] for itemID in itemIDs])
      _report(M, sourceFields)

    else:

      # loads the specified dataset
      features = loadAudioFeatures(sourcepath + [dataset], sourcefile, sourceFields)
      numOfItems    = len(features)
      numOfFeatures = len(sourceFields)

      # applies standardisation to the vector representation (using scores)
      tsprint('-- standardising feature vectors')
      M, itemIDs = _standardise(features)
      features = {itemIDs[i]: M[i] for i in range(len(itemIDs))}
      _report(M, sourceFields)

  return (features, numOfItems, numOfFeatures)

def loadAudioFeatures(sourcepath, featureFile, featureFields):
  """
    # the file must follow the structure of the Spotify's Audio Features dataset
    # https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks
  """

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
  tsprint('-- reading the file')
  df = read_csv(os.path.join(*sourcepath, featureFile))

  # parses the content to build the dictionary of features[itemID] -> feature vector
  tsprint('-- converting content into feature vectors')
  features = {}
  for _, row in df.iterrows():
    itemID  = _cast(row['id'],      'id')
    name    = _cast(row['name'],    'name')
    artists = _cast(row['artists'], 'artists')
    features[itemID] = np.array([_cast(row[field], field) for field in featureFields])

  return features

def plotTwoDists(sample1, sample2, label1, label2, ci1, ci2, plotTitle, xlabel, ylabel, filename):

  # determines some elements of the diagram
  scale_lb = min(min(sample1), min(sample2))
  scale_ub = max(max(sample1), max(sample2))
  scale_grades = 200

  # instead of histograms, induces gaussian models that fit the data
  method = 'silverman'
  try:
    kde1 = stats.gaussian_kde(sample1, method)
    kde1_pattern = 'k-'
  except np.linalg.LinAlgError:
    # if singular matrix, just black out; not a good solution, so ...
    kde1 = lambda e: [0 for _ in e]
    kde1_pattern = 'k:'

  try:
    kde2 = stats.gaussian_kde(sample2, method)
    kde2_pattern = 'r-'
  except np.linalg.LinAlgError:
    kde2 = lambda e: [0 for _ in e]
    kde2_pattern = 'r:'

  # specifies the overall grid structure
  plt.xkcd()
  fig   = plt.figure()
  panel = fig.add_subplot(111)
  plt.gca().patch.set_facecolor('0.95')
  plt.title(plotTitle)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  # plots the induced distributions
  x_eval = np.linspace(scale_lb, scale_ub, num=scale_grades)
  y_kde1 = kde1(x_eval)
  y_kde2 = kde2(x_eval)
  y_max = max(max(y_kde1), max(y_kde2))
  panel.plot(x_eval, y_kde1, kde1_pattern, label=label1)
  panel.plot(x_eval, y_kde2, kde2_pattern, label=label2)

  # plots the confidence intervals
  panel.axvline(ci1.value, 0.0, y_max, color='k', linestyle=':')
  panel.axvline(ci2.value, 0.0, y_max, color='r', linestyle=':')
  panel.fill_betweenx((0, y_max), ci1.lower_bound, ci1.upper_bound, alpha=.13, color='g')
  panel.fill_betweenx((0, y_max), ci2.lower_bound, ci2.upper_bound, alpha=.13, color='g')
  panel.legend()

  plt.savefig(filename, bbox_inches = 'tight')
  plt.close(fig)

  return None

#--------------------------------------------------------------------------------------------------
# Problem-related classes - Agent, Environment, Recommender, and Researcher
#--------------------------------------------------------------------------------------------------
class Agent:
  """
  Agent model
  name (str or int) : agent's unique identifier (analogue to a participant in the study)
  threshold (float) : an idealised construct that corresponds to the degree of surprise above which:
                      (1)  participants systematically report high level of surprise, considering a 
                           particular instrument applied to some domain;
                      (2a) longer explanations are systematically preferred by participants, or
                      (2b) participants systematically explore a larger share of the available explanations

  sensibility (int) : the agent's sensibility to perceive change in surprise (i.e., relevant decimal digits)
  """

  def __init__(self, name, threshold, sensibility):

    # properties defined during instantiation (and unmodified during runtime)
    self.name        = name
    self.threshold   = threshold
    self.sensibility = sensibility

    # properties modified during runtime
    self.profile     = None   # a dictionary holding items that are known to the agent
    self.averagevec  = None   # the average of the vectors representing items known to the agent

  def update(self, itemID = None, itemVector = None):

    # adds another item to the agent's profile
    if(itemID is not None):
      self.profile[itemID] = itemVector

    # updates the agent's average vector
    self.averagevec = np.mean(list(self.profile.values()), 0)

    return None

  def query(self, itemID, itemVector, explanation1, explanation2):

    # RECOVERS the agent's shallow surprise
    s = self.surprise(itemVector)

    # parses the explanations into explanation text (which is presented to the agent)
    # and and explanation type (which is hidden from the agent)
    (text1, type1) = explanation1
    (text2, type2) = explanation2

    if(s > self.threshold):
      # if estimated surprise is above the threshold, the agent prefers longer explanations
      option = type1 if len(text1) > len(text2) else type2
    elif(s < self.threshold):
      # if estimated surprise is below the threshold, the agent prefers shorter explanations
      option = type1 if len(text1) < len(text2) else type2
    else:
      # if estimated surprise is equal to the threshold, then anything goes
      option = choice([type1, type2])

    return (itemID, option, s)

  def surprise(self, itemVector):

    # DETERMINES the 'shallow' surprise experienced by the agent from being exposed to
    # a new recommendation.
    # ** assumes that CTM is a reasonably accurate description of human cognition
    #    (https://plato.stanford.edu/entries/computational-mind/)
    # ** assumes that the model of surprise proposed by Kaminskas and Bridge reasonably
    #    describes how people in general get surprised by events perceived as being
    #    machine-generated
    # ** assumes we have a pretty darn good instrument (questionnaire) to assess surprise
    #    in this specific recommendation scenario

    # Kaminskas, M., & Bridge, D. (2014). Measuring surprise in recommender systems
    # In Proceedings of the Workshop on Recommender Systems Evaluation: Dimensions and Design
    # (Workshop Programme of the 8th ACM Conference on Recommender Systems).
    # -- equation 5 (content-based), equipped with some distance function (which may not be a metric)
    s = min([dist(itemVector, self.profile[j]) for j in self.profile])

    # adjusts the obtained value to reflect the agent's sensibility
    s = self.adjust(s)

    return s

  def adjust(self, s):
    # reduces the precision of the agent's feedback
    # ** assumes that humans have bounded ability to ascribe numerical values to
    #    experienced events
    return round(s, self.sensibility)

class Environment:
  """
  Environment model
  dataset (dict) .....: a dictionary mapping items to item vectors (numpy arrays)
  numOfAgents (int) ..: the number of agents in the environment
                        (analogue to the number of participants recruited to the user study)
  initialSeeds (list) : a non-empty list of (threshold, sensibility) tuples, to be used in
                        initialising the population of agents
  """

  def __init__(self, dataset, numOfAgents, numOfQueries, initialSeeds):

    # properties defined during instantiation (and unmodified during runtime)
    self.dataset      = dataset
    self.numOfAgents  = numOfAgents
    self.numOfQueries = numOfQueries
    self.initialSeeds = initialSeeds

    # properties modified during runtime
    self.trainp       = None  # training partition (used to train models)
    self.testp        = None  # test     partition (used to test  models)
    self.profilep     = None  # profile  partition (used to build agent's profiles)
    self.population   = None  # the collection of agents in the environment
    self.researcher   = None  # an analogue to the researcher that will analyse the results
    self.recommender  = None  # an instance of recommendation model
    self.results      = None  # results gathered during the simulation

  def run(self):

    # splits the dataset
    tsprint('-- splitting the dataset into training, test, and profile partitions')
    self.splitDataset()
    tsprint('   {0:6d} items allocated to the training partition'.format(len(self.trainp)))
    tsprint('   {0:6d} items allocated to the test     partition'.format(len(self.testp)))
    tsprint('   {0:6d} items allocated to the profile  partition'.format(len(self.profilep)))

    if('TESTSIMUL' in os.environ):
      partition = {itemID: self.dataset[itemID] for itemID in self.trainp + self.testp}
    else:
      partition = self.dataset

    # instantiates the recommender
    tsprint('-- instantiating, training, and testing a recommendation model')
    self.recommender = Recommender(self.trainp, self.testp)

    # generates the population of agents
    tsprint('-- recruiting {0} participants'.format(self.numOfAgents))
    self.recruitParticipants()

    # simulates the interaction between the agents and the environment
    # -- it goes like this ANALOGUE (following Narrative 1 in the logbook):
    #    1. A participant is recruited to participate in our user study
    #    2. The participant is asked to identify a number of songs she likes (ECO_PROFILESIZE)
    #    3. The participant is presented to an item (the recommendation) and two explanations
    #    4. The participant is asked to answer which explanation fits best the recommendation,
    #       and also to answer a set of questions devised to estimate the experienced surprise
    #       caused by the recommendation
    #    5. Steps 3 and 4 are repeated a number of times (self.numOfQueries)

    tsprint('-- taking participants to the lab to participate in the study')
    tsprint('   each participant is asked to solve {0} tasks'.format(self.numOfQueries))
    print()

    self.results = defaultdict(list)
    for agentID in self.population:

      # 1. A participant (an agent) is recruited to participate in our user study
      agent = self.population[agentID]
      tsprint('   Welcome {0: <10} Thank you so much for taking part in our study! Please sit here ...'.format(agentID + '!'))

      # 2. The participant is asked to identify a number of songs she likes
      # [insert code here] Manzato has suggested a scheme to expand this initial selection
      # ** assumes that the agent may hold different representations of items than those
      #    held by other elements of the environment (such as the recommender system)
      agent.profile = {itemID: self.dataset[itemID] for itemID in sample(self.profilep, ECO_PROFILESIZE)}
      agent.update()

      for _ in range(self.numOfQueries):

        # 3. The participant is presented to an item (the recommendation) and two explanations

        # generates a single recommendation for the current agent and estimates its 'shallow' surprise
        itemID = self.recommender.recommend(agent, partition)
        s_hat  = self.recommender.estimateSurprise(agent, self.dataset[itemID])

        # generates two explanations for the last recommendation -- on shorter than the other
        sexp = self.recommender.explain(ECO_SHORTEXP)
        lexp = self.recommender.explain(ECO_LONGEXP)

        # randomises the presentation of the explanation on screen
        # -- imagine there are two slots (positions) in the screen where explanations are to be shown
        #    we want to show the short explanation sometimes at position 1, other times at position 2
        #    this allows us to detect (and possibly avoid) some common biases in responses to surveys
        explanations = [(sexp, ECO_SHORTEXP), (lexp, ECO_LONGEXP)]
        shuffle(explanations)

        # 4. After being presented to a single recommendation and two explanations, the participant is
        #    asked to answer which explanation fits the recommendation best, and is also asked to answer
        #    a set of questions devised to estimate the (shallow) surprise caused by the recommendation
        # ** assumes that showing a new recommendation does not imply updating the agent's profie
        (itemID, option, s) = agent.query(itemID, self.dataset[itemID], explanations[0], explanations[1])
        self.results[agentID].append((itemID, s_hat, explanations, option, s))

    print()
    tsprint('-- researcher collects and analyses the results')
    researcher = Researcher()
    researcher.collectResults(self.results)
    researcher.tabulateResults()
    researcher.testHypothesis1()

    return None

  def splitDataset(self):

    if('TESTSIMUL' in os.environ):

      itemIDs = sorted(self.dataset)
      sizeProfilep  = int(self.numOfAgents * ECO_PROFILESIZE * ECO_PROFILEVAR)
      sizeTestp     = int(ECO_SPLIT * (len(itemIDs) - sizeProfilep))
      sizeTrainp    = len(itemIDs) - sizeTestp - sizeProfilep

      self.trainp   = itemIDs[0: sizeTrainp]
      self.testp    = itemIDs[sizeTrainp: sizeTrainp + sizeTestp]
      self.profilep = itemIDs[sizeTrainp + sizeTestp: sizeTrainp + sizeTestp + sizeProfilep]

    else:

      # creates a list with all itemIDs in the dataset
      itemIDs = list(self.dataset)
      shuffle(itemIDs)

      # allocates some items in the test partition
      splitPosition = int(len(itemIDs) * ECO_SPLIT)
      self.testp = itemIDs[0: splitPosition]

      # allocates the remaining items to the training partition
      self.trainp = itemIDs[splitPosition:]

      # allocates some items to be used in profile building
      # (profiles may contain items that have been allocated to any of the previous partitions)
      self.profilep = sample(itemIDs, int(self.numOfAgents * ECO_PROFILESIZE * ECO_PROFILEVAR))

    return None

  def recruitParticipants(self):

    # generates the population of agents (analogue to participants)
    # note that agents are instantiated without a profile
    # -- this is obtained later, during the study
    self.population = {}
    names = ECO_HUMAN_NAMES
    for i in range(self.numOfAgents):
      agentID = names[i]
      (threshold, sensibility) = choice(self.initialSeeds)
      self.population[agentID] = Agent(agentID, threshold, sensibility)

    return None

class Recommender:

  def __init__(self, trainp, testp):

    # properties defined during instantiation (and unmodified during runtime)
    self.trainp = trainp
    self.testp  = testp

    # properties modified during runtime
    self.lastrec = None
    self.history = defaultdict(list)

    # trains and tests the performance of the recommender
    self.train()
    self.test()

  def train(self):

    # for now, our recommender does not require training (it is a kNN-like model)
    # [insert code here] we may want to change this later
    return None

  def test(self):

    # for now, our recommender does note require testing
    # [insert code here] we may want to change this later
    return None

  def recommend(self, agent, partition):

    # [insert code here]
    # this is a very lazy recommender! what does it do? well, it goes like this:
    # -- 1. it draws a very large, random sample of items from the dataset (ECO_LARGESAMPLE)
    #    2. it removes any items that are known to the agent or have been previously
    #       recommended to her
    #    3. it selects a single item from the sample:
    #       the one whose vector is the nearest to the agent's average vector

    # 1. it draws a very large, random sample of items from the dataset
    largeSample = sample(list(partition), ECO_LARGESAMPLE)

    # 2. it removes any items that are known to the agent or have been previously recommended to her
    largeSample = [itemID for itemID in largeSample if itemID not in agent.profile]
    largeSample = [itemID for itemID in largeSample if itemID not in self.history[agent.name]]

    # 3. it selects a single item from the sample
    orderedSample = [(itemID, dist(partition[itemID], agent.averagevec)) for itemID in largeSample]
    orderedSample.sort(key = lambda e: e[1])
    itemID = orderedSample[0][0]

    # performs some bookkeeping
    self.lastrec = (agent.name, itemID)
    self.history[agent.name].append(itemID)

    return itemID

  def estimateSurprise(self, agent, itemVector):

    # estimates the 'shallow' surprise of an agent, i.e., the surprise assessed before the agent
    # follows a recommendation, which is presumably different from the surprise assessed after
    # the agent follows the recommendation, i.e., after actually watching the recommended movie,
    # or listening to the recommended song, or reading the recommended book, etc.
    # uses a model proposed by Kaminskas and Bridge in 2014.

    return min([dist(itemVector, agent.profile[j]) for j in agent.profile])

  def explain(self, option):

    # [insert code here]
    # for now, the explanation for the last recommendation ignores the last recommendation
    # in fact, they are hardcoded, and being only conditioned on the "option" parameter
    if(option == ECO_SHORTEXP):
      explanation = 'This is a short explanation.'
    elif(option == ECO_LONGEXP):
      explanation = 'Well, in the beginning, the Earth was void and without form, ...'
    else:
      raise ValueError

    return explanation

class Researcher:

  def __init__(self):

    # properties modified during runtime
    self.rawResults  = None

    # properties related to testing the hypothesis 1
    self.data4Hypothesis1 = None
    self.CIShort = None
    self.CILong  = None

  def collectResults(self, rawResults):
    self.rawResults = rawResults
    return None

  def tabulateResults(self):

    # organises the raw results into a tabular form,
    # and saves the data in a csv file
    mask = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'
    header  = mask.format('Participant',
                          'Item ID',
                          'Estimated Surprise',
                          'Explanation 1',
                          'Explanation 2',
                          'Reported Surprise',
                          'Preferred Explanation')
    content = [header]

    for agentID in self.rawResults:
      for (itemID, s_hat, explanations, option, s) in self.rawResults[agentID]:
        buffer = mask.format(agentID,
                             itemID,
                             s_hat,
                             explanations[0][1],
                             explanations[1][1],
                             s,
                             option)

        content.append(buffer)

    saveAsText('\n'.join(content), 'rawResults.csv')

    return None

  def overlapCI(self, ci1, ci2):

    # checks if two confidence intervals overlap
    lb = max(ci1.lower_bound, ci2.lower_bound)
    ub = min(ci1.upper_bound, ci2.upper_bound)
    return((ub - lb) >= 0.0)

  def testHypothesis1(self):

    print()
    tsprint('Testing the Hypothesis 1')
    tsprint('-- association of low  reported surprise with preference for short explanations')
    tsprint('-- association of high reported surprise with preference for long  explanations')

    # prepares the raw results to test the first hypothesis
    self.data4Hypothesis1 = defaultdict(list)
    for agentID in self.rawResults:
      for (itemID, s_hat, explanations, option, s) in self.rawResults[agentID]:
        self.data4Hypothesis1[option].append(s) # uses the agent's reported surprise,
                                                # not the one estimated by the recommender

    conclusive = True
    filename = 'hypothesis1.jpg'

    # estimates the confidence interval of surprise associated with short explanations
    tsprint('-- 95% confidence interval of reported surprise associated to:')
    sample1 = np.array(self.data4Hypothesis1[ECO_SHORTEXP])
    if(sample1.size >= ECO_PROFILESIZE):
      self.CIShort = bs.bootstrap(sample1, stat_func=bs_stats.mean)
      tsprint(('   short explanations: [{0}, {1}], average {2}, sample size is {3}').format(self.CIShort.lower_bound, self.CIShort.upper_bound, self.CIShort.value, len(sample1)))
    else:
      conclusive = False
      tsprint('   short explanations: cannot be estimated because the sample has only {0} elements'.format(len(sample1)))
      tsprint('   -- maybe the surprise threshold has been poorly specified?')

    # estimates the confidence interval of surprise associated with long explanations
    sample2 = np.array(self.data4Hypothesis1[ECO_LONGEXP])
    if(sample2.size >= ECO_PROFILESIZE):
      self.CILong = bs.bootstrap(sample2, stat_func=bs_stats.mean)
      tsprint('   long  explanations: [{0}, {1}], average {2}, sample size is {3}'.format(self.CILong.lower_bound, self.CILong.upper_bound, self.CILong.value, len(sample2)))
    else:
      conclusive = False
      tsprint('   long  explanations: cannot be estimated because the sample has only {0} elements'.format(len(sample2)))
      tsprint('   -- maybe the surprise threshold has been poorly specified?')

    # assesses the overlap between the intervals
    # ** assumes that if intervals do not overlap, the hypothesis is     supported by the data
    # ** assumes that if intervals do     overlap, the hypothesis is not supported by the data
    print()
    if(conclusive):
      if(self.overlapCI(self.CIShort, self.CILong)):
        tsprint('-- THE INTERVALS OVERLAP ** the researcher assumes that the data do not support the hypothesis')
      else:
        tsprint('-- THE INTERVALS DO NOT OVERLAP ** the researcher assumes that the data support the hypothesis')

      # plots the distribution of reported surprise conditional on preference for explanation
      plotTwoDists(sample1, sample2,
                   'Short', 'Long',
                   self.CIShort, self.CILong,
                   'Distribution of reported surprise\nconditional on preferred explanation type',
                   'Reported surprise',
                   'Frequency',
                   filename)

    else:
      tsprint('-- THE STUDY WAS INCONCLUSIVE REGARDING THE HYPOTHESIS ** the researcher cries alone in the dark. so sad.')
      if(os.path.exists(filename)):
        os.remove(filename)

    return None

def fiatLux(numOfAgents, numOfQueries, initialSeeds, datasetParams):

  # initialises random number generators
  # (to control variance between repeated essays)
  seed(ECO_SEED)
  np.random.seed(ECO_SEED)

  # [insert code here]
  # for now, let's use a mocked dataset, but we want to load the Spotify dataset
  print()
  tsprint('Loading the [{0}] dataset'.format(datasetParams[0]))
  (dataset, numOfItems, numOfFeatures) = getDataset(datasetParams, numOfAgents)
  tsprint('-- dataset has {0} items; each item has {1} features.'.format(numOfItems, numOfFeatures))

  print()
  tsprint('Instantiating the simulation environment')
  environment  = Environment(dataset, numOfAgents, numOfQueries, initialSeeds)

  tsprint('Running the simulation of the user study with {0} participants.'.format(numOfAgents))
  environment.run()

  # saves the environment object for manual inspection
  serialise(dataset,       'dataset')
  serialise((environment), 'environment')

  print()
  tsprint('End of simulation.')

  saveLog('config.log')

if __name__ == "__main__":

  numOfAgents   = int(sys.argv[1])
  numOfQueries  = int(sys.argv[2])

  try:
    mockDataset = (sys.argv[3] == 'mock')
  except IndexError:
    mockDataset = False

  # [insert code here]
  # we may want to check if the parameters will lead to a successful simulation

  # a list with (threshold, sensibility) tuples
  # -- a list with a single tuple means that all agents will be instantiated with the same parameters
  #(dist, initialSeeds) = (cosdist,       [(0.2, 1)])
  (dist, initialSeeds) = (euclideandist, [(2.2, 0)])

  # this is used to test the code in controlled setting
  if('TESTSIMUL' in os.environ):
    print()
    tsprint('--------------------------------- SIMULATION RUNNING IN TEST MODE ---------------------------------')
    dist = cosdist
    initialSeeds = [(0.5, 1)]
    mockDataset = True

  # determines the dataset used in the simulation
  if(mockDataset):
    dataset      = 'mocked'
    sourcepath   = None
    sourcefile   = None
    sourceFields = None

  else:
    dataset      = 'spotify'
    sourcepath   = [getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'datasets']
    sourcefile   = 'tracks.csv'
    sourceFields = ['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
                    'popularity', 'valence', 'release_date']

  datasetParams = (dataset, sourcepath, sourcefile, sourceFields)
  fiatLux(numOfAgents, numOfQueries, initialSeeds, datasetParams)
