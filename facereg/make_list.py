import os
import random
import pickle
rootpath = os.getcwd()

def get_dict(sp):
    sp_path = os.path.join(rootpath, sp)
    persons = os.listdir(sp_path)

    dictx={}
    for id, p in enumerate(persons):
        person_path = os.path.join(sp_path, p)
        imgs = os.listdir(person_path)
        for im in imgs:
            im = os.path.join(person_path, im)
            if not dictx.has_key(p):
                dictx[p] = []
            dictx[p].append(im)
    return dictx
    
trainset = get_dict('train-align2')
testset = get_dict('test-align2')

output = open('train-align2.pkl', 'wb')
pickle.dump(trainset, output)
output.close()
output = open('test-align2.pkl', 'wb')
pickle.dump(testset, output)
output.close()



dictall = dict(trainset)
dictall.update(testset)
print(len(dictall))

trainfid = open('train-align2.txt', 'w')
testfid = open('test-align2.txt', 'w')

for id, p in enumerate(dictall):
    sp = len(dictall[p])-10 #int(0.9 * len(dictall[p]))
    for im in dictall[p][:sp-1]:
        trainfid.write('%s %s\n' % (im, id))
    for im in dictall[p][sp:]:
        testfid.write('%s %s\n' % (im, id))

trainfid.close()
testfid.close()
