# Script de Preparación de Datos
###################################

# Importación de librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import sys, warnings, os

# Para ignorar advertencias
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# Configuración de pandas para mostrar todas las columnas
pd.set_option("display.max_columns", None, "display.max_rows", None)

# make_dataset: Script de Preparación de Datos
def read_file_csv(filename):
    ufc_master_ds = pd.read_csv(os.path.join('../data/draw/', filename)).set_index('Winner')
    print(filename, ' cargado correctamente')
    return ufc_master_ds
def data_preparation(ufc_master_ds):
    label = ufc_master_ds['Winner']
    X = ufc_master_ds.drop(['Winner'], axis =1)
    ufc_master_ds['draw_diff'] = (ufc_master_ds['BlueDraws']-ufc_master_ds['RedDraws'])
    ufc_master_ds['avg_sig_str_pct_diff'] = (ufc_master_ds['BlueAvgSigStrPct']-ufc_master_ds['RedAvgSigStrPct'])
    ufc_master_ds['avg_TD_pct_diff'] = (ufc_master_ds['BlueAvgTDPct']-ufc_master_ds['RedAvgTDPct'])
    ufc_master_ds['win_by_Decision_Majority_diff'] = (ufc_master_ds['BlueWinsByDecisionMajority']-ufc_master_ds['RedWinsByDecisionMajority'])
    ufc_master_ds['win_by_Decision_Split_diff'] = (ufc_master_ds['BlueWinsByDecisionSplit']-ufc_master_ds['RedWinsByDecisionSplit'])
    ufc_master_ds['win_by_Decision_Unanimous_diff'] = (ufc_master_ds['BlueWinsByDecisionUnanimous']-ufc_master_ds['RedWinsByDecisionUnanimous'])
    ufc_master_ds['win_by_TKO_Doctor_Stoppage_diff'] = (ufc_master_ds['BlueWinsByTKODoctorStoppage']-ufc_master_ds['RedWinsByTKODoctorStoppage'])
    ufc_master_ds['odds_diff'] = (ufc_master_ds['BlueOdds']-ufc_master_ds['RedOdds'])
    ufc_master_ds['ev_diff'] = (ufc_master_ds['BlueExpectedValue']-ufc_master_ds['RedExpectedValue'])

    #Dropping variables
    var_drop = [
    'BlueOdds',
    'RedOdds',
    'BlueCurrentLoseStreak', 'RedCurrentLoseStreak',
    'BlueCurrentWinStreak', 'RedCurrentWinStreak',
    'BlueLongestWinStreak', 'RedLongestWinStreak',
    'BlueWins', 'RedWins',
    'BlueLosses', 'RedLosses',
    'BlueTotalRoundsFought', 'RedTotalRoundsFought',
    'BlueTotalTitleBouts', 'RedTotalTitleBouts',
    'BlueWinsByKO', 'RedWinsByKO',
    'BlueWinsBySubmission', 'RedWinsBySubmission',
    'BlueHeightCms', 'RedHeightCms',
    'BlueReachCms', 'RedReachCms',
    'BlueAge', 'RedAge',
    'BlueAvgSigStrLanded', 'RedAvgSigStrLanded',
    'BlueAvgSubAtt', 'RedAvgSubAtt',
    'BlueAvgTDLanded', 'RedAvgTDLanded',
    'BlueDraws','BlueAvgSigStrPct','BlueAvgTDPct','BlueWinsByDecisionMajority','BlueWinsByDecisionSplit','BlueWinsByDecisionUnanimous','BlueWinsByTKODoctorStoppage',
    'RedDraws','RedAvgSigStrPct','RedAvgTDPct','RedWinsByDecisionMajority','RedWinsByDecisionSplit','RedWinsByDecisionUnanimous','RedWinsByTKODoctorStoppage']

    ufc_master_ds.drop(var_drop, axis=1, inplace = True)

  # Eliminar columnas comunes
    comm_drop = [
    'Date','Location','Country','WeightClass','Gender','NumberOfRounds','EmptyArena','Finish','FinishDetails','FinishRound','FinishRoundTime','TotalFightTimeSecs','BlueWeightLbs','RedWeightLbs'
    ]

    ufc_master_ds.drop(comm_drop, axis=1, inplace = True)

    #It has one spelling mistake
    ufc_master_ds['BlueStance'].loc[ufc_master_ds['BlueStance']=='Switch '] = 'Switch'

    stance = ['BlueStance', 'RedStance']

    for x in stance:
      ufc_master_ds[x] = [4 if st == 'Orthodox'
                            else 3 if st == 'Southpaw'
                            else 2 if st == 'Switch'
                            else 1 for st in ufc_master_ds[x]]
    #using -1 and 1 for both red and blue so there is no misunderstanding that one variable is better than the other
    ufc_master_ds['BetterRank'] = [-1 if rank == 'Red'
                                  else 1 if rank == 'Blue'
                                  else 0 for rank in ufc_master_ds['BetterRank']]

    ufc_master_ds['TitleBout'] = [1 if tb==True else 0 for tb in ufc_master_ds['TitleBout']]

    ufc_master_ds['Stance_diff'] = (ufc_master_ds['BlueStance'] - ufc_master_ds['RedStance'])
    ufc_master_ds.drop(stance, axis = 1, inplace = True)

    ufc_master_ds['Winner'] = [1 if winner == 'Red' else 0 for winner in ufc_master_ds.Winner]
    ufc_master_ds.drop(ufc_master_ds.loc[:,'BMatchWCRank':'BPFPRank'], axis=1, inplace = True)

    # X = ufc_master_ds.drop(['Winner'], axis=1)

    label = ufc_master_ds.Winner
    ufc_master_ds.drop(['Winner'], axis=1, inplace = True)

    

    # Definir columnas categóricas y numéricas
    cat_col = ['RedFighter', 'BlueFighter']

    # Codificación de las columnas categóricas
    enc = LabelEncoder()
    for i in ufc_master_ds[cat_col]:
        ufc_master_ds[i] = enc.fit_transform(ufc_master_ds[i])
    return ufc_master_ds

# Exportamos la matriz
def data_exporting(ufc_master_ds,filename):
    ufc_master_ds.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')



def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('ufc-master_final.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1,'ufc-master.csv')

if __name__ == "__main__":
    main()

