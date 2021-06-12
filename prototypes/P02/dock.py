from os.path import join
from sharedDefs import getMountedOn, deserialise, report, saveAsText, dict2text, distance
from sharedDefs import tsprint, headerfy

def main():

  featureFields = ['acousticness',     'danceability', 'duration_ms', 'energy',   'explicit',
                   'instrumentalness', 'key',          'liveness',    'loudness', 'mode',
                   'speechiness',      'tempo',        'popularity',  'valence',  'release_date']

  sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']
  targetpath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']

  # loads preprocessed data
  tsprint('Loading preprocessed data')
  features = deserialise(join(*sourcepath, 'features'))
  interior = deserialise(join(*sourcepath, 'interior'))
  hull     = deserialise(join(*sourcepath, 'hull'))
  (allPopIDs, allRegIDs, popIDs, regIDs) = deserialise(join(*sourcepath, 'samples'))

  idx_popularity = featureFields.index('popularity')
  W = [features[popIDs[i]] for i in hull.vertices]

  # estimates the relationship between popularity and surprise
  tsprint('Estimating the relationship between popularity and upper bound surprise')

  mask    = '{0}\t{1}\t{2:6.3f}\t{3:6.3f}'
  header  = headerfy(mask).format('Item ID', 'Position', 'Popularity', 'Surprise UB')
  content = [header]

  for i in range(len(regIDs)):
    itemID = regIDs[i]
    v = features[itemID]
    popularity = features[itemID][idx_popularity]
    surpriseub = max([distance(v, w) for w in W])
    position   = 'interior' if interior[i] else 'exterior'
    content.append(mask.format(itemID, position, popularity, surpriseub))

  tsprint('Job completed.')
  saveAsText('\n'.join(content), join(*targetpath, 'temp.csv'))

if __name__ == "__main__":

  main()
