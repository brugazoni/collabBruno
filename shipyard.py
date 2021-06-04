import sharedDefs as ud

from os.path    import join

from sharedDefs import getMountedOn, deserialise, tsprint, dict2text, saveAsText, serialise, saveLog
from sharedDefs import mapURL2ID

def main():

  sourcepath=[getMountedOn(), 'Task Stage', 'Task - collabBruno', 'collabBruno', 'results',  'spotify']

  tsprint('Loading serialised data')
  features=deserialise(join(*sourcepath, 'features'))
  id2name=deserialise(join(*sourcepath, 'id2name'))
  name2id=deserialise(join(*sourcepath, 'name2id'))
  rankings=deserialise(join(*sourcepath, 'rankings'))
  timeline=deserialise(join(*sourcepath, 'timeline'))
  songs=deserialise(join(*sourcepath, 'songs'))

  # builds the relationship between the datasets
  # -- xxx is a map between WDSR.id and D600k.id
  ud.LogBuffer = []

  tsprint('Linking songs reported in daily rankings to their feature vectors')
  (url2id, failures, cases, samples) = mapURL2ID(songs, id2name, name2id)
  tsprint('-- {0} popular songs have been identified.'.format(len(songs)))
  tsprint('-- {0} popular songs were linked to their feature vector'.format(len(url2id)))
  tsprint('-- {0} popular songs remain unlinked'.format(len(failures)))
  tsprint('-- {0}'.format(failures), verbose=False)

  serialise(url2id, join(*sourcepath, 'url2id'))

  saveAsText(dict2text(cases),   join(*sourcepath, 'cases.csv'))
  saveAsText(dict2text(samples), join(*sourcepath, 'samples.csv'))
  saveLog(join(*sourcepath, 'config.log'))

if __name__ == "__main__":

  main()
