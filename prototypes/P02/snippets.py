from os.path import join
from collections import defaultdict
from sharedDefs import getMountedOn, deserialise, report, saveAsText, dict2text, distance

def main():

  sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']
  targetpath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']

  features = deserialise(join(*sourcepath, 'features'))
  id2name  = deserialise(join(*sourcepath, 'id2name'))
  name2id  = deserialise(join(*sourcepath, 'name2id'))
  rankings = deserialise(join(*sourcepath, 'rankings'))
  timeline = deserialise(join(*sourcepath, 'timeline'))
  songs    = deserialise(join(*sourcepath, 'songs'))
  url2id   = deserialise(join(*sourcepath, 'url2id'))
  hull     = deserialise(join(*sourcepath, 'hull'))

  urlID='0Mh4id8WvrTFOB9RiVitrB'
  print(report(urlID, songs, url2id, id2name))

if __name__ == "__main__":

  main()
