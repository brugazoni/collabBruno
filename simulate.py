# for tomorrow: fix xxx; save results as an excel table
"""

  simulate: simulates a user study to clarify the relationship between surprise
            and preferred explanation (shorter or longer) in recommender systems
            using a user model (Agent), and performs a hypothesis test to check
            if the obtained results support the existence of the relationship
            (in this very, very idealised experimental conditions)

  Example: python simulate.py <number of items>, <number of features>, <number of agents>

"""

import sys
import pickle
import codecs
import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from collections import defaultdict
from random      import seed, sample, choice, shuffle

# parameters used in generating and handling the mocked dataset
ECO_SEED  = 23
ECO_MU    = 0.5
ECO_SD    = 0.25
ECO_SPLIT = 0.2    # the fraction of the dataset that is allocated to the test partition
                   # the remaining items are allocated to the test partition

ECO_PROFILEVAR = 2 # used to specify the size of the profile partition
                   # indirectly controls the overlap among agents' profiles:
                   # the larger the value, the smaller the overlap
                   # (with overlap being measured with the Jaccard distance for sets)

# parameters used to specify the user study
ECO_PROFILESIZE  = 10
ECO_NUMOFQUERIES = 10
ECO_LARGESAMPLE  = 10

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

#--------------------------------------------------------------------------------------------------
# Problem-related definitions - Inspection helpers
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
# Problem-related definitions
#--------------------------------------------------------------------------------------------------

def cosdist(v, w):
  # positive-shifted cosine distance (returns values in the real interval [0, 1])
  # assumes v and w are numpy arrays with similar number of elements
  return (1 - v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w))) / 2

def generateDataset(numOfItems, numOfFeatures):
  # generates a mocked dataset containing "numOfItems" items,
  # each described by "numOfFeatures" features
  # *** assumes the features represent counting variables
  # *** assumes the features can be approximately described by normal distributions
  dataset = {itemID: np.random.normal(ECO_MU, ECO_SD, numOfFeatures) for itemID in range(numOfItems)}
  return dataset

#--------------------------------------------------------------------------------------------------
# Problem-related classes - Agent, Environment, Recommender, and Researcher
#--------------------------------------------------------------------------------------------------
class Agent:
  """
  Agent model
  name (str or int) ..: agent's unique identifier
  threshold (float) : a measure of surprise above which longer explanations will be preferred
  sensibility (int) : the agent sensibility to variation in surprise, as the number of relevant decimal digits
  """

  def __init__(self, name, threshold, sensibility):

    # properties defined during instantiation (and unmodified during runtime)
    self.name        = name
    self.threshold   = threshold
    self.sensibility = sensibility

    # properties modified during runtime
    self.profile     = None
    self.averagevec  = None

  def update(self, itemID = None, itemVector = None):

    # adds another item to the agent's profile
    if(itemID is not None):
      self.profile[itemID] = itemVector

    # updates the agent's average vector
    self.averagevec = np.mean(list(self.profile.values()), 0)

  def query(self, itemID, itemVector, explanation1, explanation2):

    # estimates the agent's surprise caused by exposing it to the recommended item
    s_hat = self.surprise(itemVector)

    if(s_hat > self.threshold):
      # if estimated surprise is above the threshold, the agent prefers longer explanations
      option = explanation1 if len(explanation1) > len(explanation2) else explanation2
    else:
      # if estimated surprise is below the threshold, the agent prefers shorter explanations
      option = explanation1 if len(explanation1) < len(explanation2) else explanation2

    return (itemID, option, s_hat)

  def surprise(self, itemVector):

    # estimates the surprise of the recommendation using Kaminskas and Bridge model
    #
    #   Kaminskas, M., & Bridge, D. (2014, October). Measuring surprise in recommender
    #   systems. In Proceedings of the Workshop on Recommender Systems Evaluation:
    #   Dimensions and Design (Workshop Programme of the 8th ACM Conference on Recommender
    #   Systems).
    #
    #   -- equation 5 (content-based), equipped with positive-shifted cosine distance
    s_hat = min([cosdist(itemVector, self.profile[j]) for j in self.profile])

    # adjusts the obtained estimate for agent's sensibility
    s_hat = self.adjust(s_hat)

    return s_hat

  def adjust(self, s_hat):
    return round(s_hat, self.sensibility)

class Environment:
  """
  Environment model
  dataset (dict) .....: a dataset with items described by vectors (numpy arrays)
  numOfAgents (int) ..: the number of agents in the environment
                        (analogous to the number of participants recruited to a user study)
  initialSeeds (list) : a non-empty list of (threshold, sensibility) tuples, to be used in
                        initialising the population of agents
  """

  def __init__(self, dataset, numOfAgents, initialSeeds):

    # properties defined during instantiation (and unmodified during runtime)
    self.dataset      = dataset
    self.numOfAgents  = numOfAgents
    self.initialSeeds = initialSeeds

    # properties modified during runtime
    self.trainp       = None  # training partition (used to train models)
    self.testp        = None  # test     partition (used to test  models)
    self.profilep     = None  # profile  partition (used to build agent's profiles)
    self.population   = None  # the collection of agents in the environment
    self.recommender  = None  # an instance of recommendation model
    self.results      = None  # results gathered during the simulation

  def run(self):

    # splits the dataset
    print('-- splitting the dataset into training, test, and profile partitions')
    self.splitDataset()

    # instantiates the recommender
    print('-- instantiating, training, and testing a recommendation model')
    self.recommender = Recommender(self.trainp, self.testp)

    # generates the population of agents
    print('-- recruiting {0} participants'.format(self.numOfAgents))
    self.generatePopulation()

    # simulates the interaction between the agents and the environment
    # -- it goes like this analogue:
    #    1. A participant (an agent) is recruited to participate in our user study
    #    2. The participant is asked to identify a number of songs she likes (ECO_PROFILESIZE)
    #    3. The participant is presented to an item (the recommendation) and two explanations
    #    4. The participant is asked to answer which explanation fits best the recommendation,
    #       and also to answer a set of questions devised to estimate the experienced surprise
    #       caused by the recommendation
    #    5. Steps 3 and 4 are repeated a number of times (ECO_NUMOFQUERIES)

    print('-- taking participants to the lab to participate in the study')
    print('   each participant is asked to solve {0} tasks'.format(ECO_NUMOFQUERIES))
    print()

    self.results = defaultdict(list)
    for agentID in self.population:

      # 1. A participant (an agent) is recruited to participate in our user study
      agent = self.population[agentID]
      print('   Welcome {0: <10}! Thank you so much for taking part in our study! Please sit here ...'.format(agentID))

      # 2. The participant is asked to identify a number of songs she likes
      agent.profile = {itemID: self.dataset[itemID] for itemID in sample(self.profilep, ECO_PROFILESIZE)}
      agent.update()

      for _ in range(ECO_NUMOFQUERIES):

        # 3. The participant is presented to an item (the recommendation) and two explanations

        # generates a single recommendation for the current agent
        itemID = self.recommender.recommend(agent, self.dataset)

        # generates two explanations for the last recommendation -- on shorter than the other
        sexp = self.recommender.explain(ECO_SHORTEXP)
        lexp = self.recommender.explain(ECO_LONGEXP)

        # randomises the presentation of the explanation on screen
        # -- imagine there are two slots (positions) in the screen where explanations are to be shown
        #    we want to show the short explanation sometimes in position 1, other times in position 2
        #    we want this to detect (and possibly detect) some common biases in survey responses
        explanations = [(sexp, ECO_SHORTEXP), (lexp, ECO_LONGEXP)]
        shuffle(explanations)

        #    4. The participant is asked to answer which explanation fits best the recommendation,
        #       and also to answer a set of questions devised to estimate the experienced surprise
        #       caused by the recommendation

        (itemID, option, s_hat) = agent.query(itemID, self.dataset[itemID], explanations[0], explanations[1])
        self.results[agentID].append((itemID, option, s_hat))

  def splitDataset(self):

    # creates a list with all itemIDs in the dataset
    itemIDs = list(self.dataset)

    # allocates some items in the test partition
    self.testp = sample(itemIDs, int(len(itemIDs) * ECO_SPLIT))

    # allocates the remaining items to the training partition
    self.trainp = [itemID for itemID in itemIDs if itemID not in self.testp]

    # allocates some items to be used in profile building
    # (profiles may contain items that have been allocated to any of the previous partitions)
    self.profilep = sample(itemIDs, int(self.numOfAgents * ECO_PROFILESIZE * ECO_PROFILEVAR))

  def generatePopulation(self):

    self.population = {}
    names = ECO_HUMAN_NAMES
    for i in range(self.numOfAgents):
      agentID = names[i]
      (threshold, sensibility) = choice(self.initialSeeds)
      self.population[agentID] = Agent(agentID, threshold, sensibility)

class Recommender:

  def __init__(self, trainp, testp):

    # properties defined during instantiation (and unmodified during runtime)
    self.trainp = trainp
    self.testp  = testp

    # properties modified during runtime
    self.lastrec = None

    # trains and tests the performance of the recommender
    self.train()
    self.test()

  def train(self):

    # for now, let's suppose our recommender does not require training
    # [insert code here]
    None

  def test(self):

    # for now, let's suppose our recommender does note require testing
    # [insert code here]
    None

  def recommend(self, agent, dataset):

    # [insert code here]
    # this is a very lazy recommender!
    # what does it do? well, it goes like this:
    # -- 1. it draws a very large, random sample of items from the dataset
    #    2. it removes any items that are known to the agent
    #    3. it selects a single item from the sample:
    #       the one whose vector is nearly in the same direction of the average profile vector

    largeSample = sample(list(dataset), ECO_LARGESAMPLE)
    largeSample = [itemID for itemID in largeSample if itemID not in agent.profile]
    L1 = [(itemID, cosdist(dataset[itemID], agent.averagevec)) for itemID in largeSample]
    L1.sort(key = lambda e: e[1])
    self.lastrec = L1[0][0]

    return self.lastrec

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

  def __init__(self, rawResults):

    # properties defined during instantiation (and unmodified during runtime)
    self.rawResults = rawResults

    # properties related to testing the hypothesis 1
    self.data4Hypothesis1 = None
    self.CIShort = None
    self.CILong  = None

  def overlapCI(self, ci1, ci2):

    # checks if two confidence intervals overlap
    lb = max(ci1.lower_bound, ci2.lower_bound)
    ub = min(ci1.upper_bound, ci2.upper_bound)
    return((ub - lb) >= 0.0)

  def testHypothesis1(self):

    print('Testing the Hypothesis 1')
    print('-- association of low  estimated surprise and preference for short explanations')
    print('-- association of high estimated surprise and preference for long  explanations')

    # prepares the raw results to test the first hypothesis
    self.data4Hypothesis1 = defaultdict(list)
    for agentID in self.rawResults:
      for (itemID, option, s_hat) in self.rawResults[agentID]:
        (_, choiceOfExplanation) = option
        self.data4Hypothesis1[choiceOfExplanation].append(s_hat)

    # computes the confidence interval of surprise for short explanations
    sample1 = self.data4Hypothesis1[ECO_SHORTEXP]
    self.CIShort = bs.bootstrap(np.array(sample1), stat_func=bs_stats.sum)
    print('-- 95% confidence interval of surprise associated to:')
    print('   short explanations is [{0}, {1}], average {2}'.format(self.CIShort.lower_bound, self.CIShort.upper_bound, self.CIShort.value))

    # computes the confidence interval of surprise for long explanations
    sample2 = self.data4Hypothesis1[ECO_LONGEXP]
    self.CILong = bs.bootstrap(np.array(sample2), stat_func=bs_stats.sum)
    print('   long  explanations is [{0}, {1}], average {2}'.format(self.CILong.lower_bound, self.CILong.upper_bound, self.CILong.value))

    # assesses the overlap between the intervals
    # ** assumes that if intervals do not overlap, the hypothesis is     supported by the data
    # ** assumes that if intervals do     overlap, the hypothesis is not supported by the data
    if(self.overlapCI(self.CIShort, self.CILong)):
      print('-- THE INTERVALS OVERLAP ** assume that the data do not support the hypothesis 1')
    else:
      print('-- THE INTERVALS DO NOT OVERLAP ** assume that the data support the hypothesis 1')

    return None # xxx include this in all methods, except constructors

def main(numOfItems, numOfFeatures, numOfAgents):

  # initialises random number generator
  # (to control variance between repeated essays)
  seed(ECO_SEED)

  # [insert code here]
  # for now, let's use a mocked dataset, but
  # it must be replaced to load the Spotify dataset
  print()
  print('Generating the mocked dataset with {0} items and {1} features.'.format(numOfItems, numOfFeatures))
  dataset = generateDataset(numOfItems, numOfFeatures)

  print('Instantiating the simulation environment.')
  initialSeeds = [(0.05, 8)] # a list with a single tuple means that all agents will be instantiated
                            # with the same parameters for threshold and sensibility
  environment  = Environment(dataset, numOfAgents, initialSeeds)

  print()
  print('Running the simulation of the user study with {0} participants.'.format(numOfAgents))
  environment.run()

  print()
  print('Analysing the results')
  researcher = Researcher(environment.results)
  researcher.testHypothesis1()

  # saves the environment object for manual inspection
  serialise((environment), 'environment')

  print()
  print('Done.')

if __name__ == "__main__":

  numOfItems    = int(sys.argv[1])
  numOfFeatures = int(sys.argv[2])
  numOfAgents   = int(sys.argv[3])

  # checks if the parameters will lead to a successful simulation
  # [insert code here]

  main(numOfItems, numOfFeatures, numOfAgents)
