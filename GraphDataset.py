import torch
import pickle
import numpy as np
import pandas as pd

from dateutil import parser
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, ListaArchivos, etiquetas, caract = 'VGGish'):
        self.caract = caract
        
        self.lista = ListaArchivos
        self.n = len(ListaArchivos)#Número de grafos a crear = número de grabadora
        self.y = etiquetas
        self.dates = ['2022-09-05 10:00:00','2022-09-06 10:00:00','2022-09-07 10:00:00','2022-09-08 10:00:00',]
        self.times = [0,15,30,45]
        self.GLOB_PATH = '/media/andrescastro/Seagate Backup Plus Drive'
    
    def __getitem__(self, index):
        archivo = index #qué archivo usar
        # print(index)
        ruta = self.lista[archivo]
        name = '_'.join(ruta.split('/'))
        print(name)
        
        if self.caract == 'VGGish':
            self.path =  self.GLOB_PATH + '/AECO/DeepFeatures_data/ReyZamuro/'
            file_path = f'{self.path}{name}_vggish.pickle'
            with open(file_path, 'rb') as handle:
                unserialized_data = pickle.load(handle)
            
            Vggtensor = unserialized_data['Data'].mean(1)            
            n_feat = 128
        
        elif self.caract == 'YAMNet':
            self.path =  self.GLOB_PATH + '/AECO/DeepFeatures_data/ReyZamuro/'
            file_path = f'{self.path}{name}_yamnet.pickle'
            with open(file_path, 'rb') as handle:
                unserialized_data = pickle.load(handle)
            
            Yamn_tensor = unserialized_data['Data'].mean(1)
            n_feat = 1024

        elif self.caract == 'PANNs':
            self.path = self.GLOB_PATH + '/AECO/DeepFeatures_data/ReyZamuro/'
            file_path = f'{self.path}{name}_panns.pickle'
            with open(file_path, 'rb') as handle:
                unserialized_data = pickle.load(handle)
            
            panns_tensor = unserialized_data['Data']
            n_feat = 2048

        elif self.caract == 'AI':
            self.path = self.GLOB_PATH + '/AECO/AcousticIndices_data/ReyZamuro/'
            file_path = f'{self.path}{name}_AIs.csv'
            df_feat = pd.read_csv(file_path, index_col='Date')
            df_feat.drop(['file'], axis=1, inplace=True)
            df_feat.index = pd.Series(df_feat.index.map(parser.parse))
            df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_feat = df_feat.fillna(df_feat.mean())

            n_feat = 60

        # print(f'{file_path=}')        
        
        df = pd.read_csv(f'{self.GLOB_PATH}/Conv1D_ecological_RZ/ZamuroCaract/CSVs/{name}.csv', index_col=0)
        df.index = pd.Series(df.index.map(parser.parse))
        # print(f'{n_feat=}')

        L = np.arange(df.shape[0])
        Data1 = torch.empty((0,24,n_feat)) 

        for i in range(3):#Tenemos tres días
            Data2 = torch.empty((0,n_feat))
            for bloque in range(24):
                d = i
                Hour = (bloque+10)%24
                if (Hour >= 0 and Hour<=9):
                    d += 1
                date = self.dates[d]
                # print(Hour)
                # print(df[(df.index.date == pd.to_datetime(date).date()) &
                #          (df.index.hour == Hour)])
                Sel = (df.index.date == pd.to_datetime(date).date()) & (df.index.hour == Hour)
                indices_datos = L[Sel]

                if self.caract == 'VGGish':
                    fila1 = Vggtensor[indices_datos,...].mean(0)
                elif self.caract == 'YAMNet':
                    fila1 = Yamn_tensor[indices_datos,...].mean(0)
                elif self.caract == 'PANNs':
                    fila1 = panns_tensor[indices_datos,...].mean(0)
                elif self.caract == 'AI':
                    fila1 = torch.tensor(df_feat.iloc[indices_datos,:].mean(0).values.astype('float'))
                    # print(df.iloc[indices_datos,:])
                
                # if (fila1.shape[0] != 4):
                #     fila1 = torch.cat((fila1[0].unsqueeze(0), fila1))

                Data2 = torch.cat((Data2,fila1.unsqueeze(0)), dim=0)
            
            Data1 = torch.cat((Data1,Data2.unsqueeze(0)), dim = 0)
        print(f'{Data1.shape=}')
        label = self.y[archivo]

        return Data1, label

    def __len__(self):
        return self.n