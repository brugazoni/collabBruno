from os.path import join
from collections import defaultdict
from sharedDefs import getMountedOn, deserialise, report, saveAsText, dict2text, distance

sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']

features = deserialise(join(*sourcepath, 'features'))
id2name  = deserialise(join(*sourcepath, 'id2name'))
name2id  = deserialise(join(*sourcepath, 'name2id'))
rankings = deserialise(join(*sourcepath, 'rankings'))
timeline = deserialise(join(*sourcepath, 'timeline'))
songs    = deserialise(join(*sourcepath, 'songs'))
url2id   = deserialise(join(*sourcepath, 'url2id'))
failures = deserialise(join(*sourcepath, 'failures'))
#hull     = deserialise(join(*sourcepath, 'hull'))

featureFields = ['acousticness',     'danceability', 'duration_ms', 'energy',   'tempo',
                 'instrumentalness', 'release_date', 'liveness',    'loudness', 'mode',
                 'speechiness',      'explicit',     'popularity',  'valence',  'key']

#urlID='0Mh4id8WvrTFOB9RiVitrB'
#print(report(urlID, songs, url2id, id2name))

#_idx_popularity = featureFields.index('popularity')

#L = sorted([(itemID, urlID, features[itemID][_idx_popularity]) for urlID in url2id for itemID in [url2id[urlID]]], key = lambda e: e[2])
