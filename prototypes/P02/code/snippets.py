import sys

from os.path import join
from collections import defaultdict
from sharedDefs import getMountedOn, deserialise, report, saveAsText, dict2text, distance

essayid  = sys.argv[1]
configid = sys.argv[2]

sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify', essayid, configid]

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

#ECO_SEED=23
#Q_ = np.vstack([features[itemID] for itemID in itemIDs])
#pca = PCA(n_components = 5, svd_solver = 'arpack', random_state = ECO_SEED)
#Q  = pca.fit_transform(Q_)

