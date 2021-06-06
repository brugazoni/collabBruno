from os.path import join
from sharedDefs import getMountedOn, deserialise, report
sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']
features=deserialise(join(*sourcepath, 'features')) 
id2name=deserialise(join(*sourcepath, 'id2name'))
name2id=deserialise(join(*sourcepath, 'name2id'))
rankings=deserialise(join(*sourcepath, 'rankings'))
timeline=deserialise(join(*sourcepath, 'timeline'))
songs=deserialise(join(*sourcepath, 'songs'))
url2id=deserialise(join(*sourcepath, 'url2id'))
urlID='0Mh4id8WvrTFOB9RiVitrB'
print(report(urlID, songs, url2id, id2name))
