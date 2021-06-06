from os.path import join
#from collections import defaultdict
from sharedDefs import getMountedOn, deserialise, report, saveAsText, dict2text, distance
from sharedDefs import tsprint, headerfy

def main():

  featureFields = ['acousticness',     'danceability', 'duration_ms', 'energy',   'explicit',
                   'instrumentalness', 'key',          'liveness',    'loudness', 'mode',
                   'speechiness',      'tempo',        'popularity',  'valence',  'release_date']

  sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']
  targetpath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']

  # loading preprocessed data
  tsprint('Loading preprocessed data')
  features = deserialise(join(*sourcepath, 'features'))
  interior = deserialise(join(*sourcepath, 'interior'))
  hull     = deserialise(join(*sourcepath, 'hull'))
  (allPopIDs, allItemIDs, popIDs, itemIDs) = deserialise(join(*sourcepath, 'samples'))

  idx_popularity = featureFields.index('popularity')
  W = [features[popIDs[i]] for i in hull.vertices]

  tsprint('Estimating the relationship between popularity and upper bound surprise')
  hull = Delaunay([hull.points[i] for i in hull.vertices])

  mask = '{0}\t{1}\t{2:6.3f}\t{3:6.3f}'
  header = headerfy(mask).format('Item ID', 'Position', 'Popularity', 'Surprise UB')
  content = [header]

  for itemID in itemIDs:
    v = features[itemID]
    popularity = features[itemID][idx_popularity]
    surpriseub = max([distance(v, w) for w in W])
    position   = 'interior' if hull.find_simplex(v) else 'exterior'
    content.append(mask.format(itemID, position, popularity, surpriseub))

  saveAsText('\n'.join(content), join(*targetpath, 'temp.csv'))

  #saveAsText(dict2text(popularity2surprise, header=['Popularity', 'Surprise UB']), join(*targetpath, 'temp.csv'))

if __name__ == "__main__":

  main()
