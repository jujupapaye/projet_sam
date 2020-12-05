from sam_io import read_wav
from librosa.feature import chroma_stft, chroma_cqt,  mfcc
from librosa.display import specshow
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np

msdi_path = '/data/msdi'

clasical_music_path = '/data/clasical_sound/'


def load_music_to_chroma(path):
    fs, x = read_wav(path)
    chroma = chroma_stft(x, fs)
    return chroma


def load_music_to_MFCC(path, der=0):
    fs, x = read_wav(path)
    x_mfcc = mfcc(x, fs)
    return x_mfcc


def show(x, type='chroma'):
    if type == 'chroma':
        specshow(x, y_axis='chroma', x_axis='time')
        plt.colorbar()
    else:
        specshow(x, x_axis='time')
        plt.colorbar()


composithors = ["beethoven", "chopin", "liszt", "mozart"]

nbre_fichiers = 120
clasical_chromas = np.zeros((nbre_fichiers, 12, 2584))  # j'ai mis ici la taille maximum des fichiers
clasical_mfcc = np.zeros((nbre_fichiers, 20, 2584))    # si un fichier est plus petit, on complète la fin par des zeros
y_comp = np.zeros(nbre_fichiers)

i = 0
for compo in range(len(composithors)):
    fichiers = glob.glob("./" + clasical_music_path + composithors[compo] + "/*")
    for file in fichiers:
        mfc = load_music_to_MFCC(file)
        chroma = load_music_to_chroma(file)
        clasical_mfcc[i, :, :mfc.shape[1]] = mfc
        clasical_chromas[i, :, :chroma.shape[1]] = chroma
        y_comp[i] = compo
        i += 1

clasical_chromas = clasical_chromas.reshape(nbre_fichiers, clasical_chromas.shape[1]*clasical_chromas.shape[2]) # on vectorise nos données, un son = un vecteur
clasical_mfcc = clasical_mfcc.reshape(nbre_fichiers, clasical_mfcc.shape[1]*clasical_mfcc.shape[2])

df_chromas = pd.DataFrame(data=clasical_chromas)
df_mfcc = pd.DataFrame(data=clasical_mfcc)
df_comp = pd.DataFrame(data=y_comp)

chromas = pd.concat([df_chromas, df_comp], axis=1, ignore_index=True)
mfccs = pd.concat([df_chromas, df_comp], axis=1, ignore_index=True)

chromas.to_csv("chromas_clasical_sound.csv")
mfccs.to_csv("mfccs_clasical_sound.csv")