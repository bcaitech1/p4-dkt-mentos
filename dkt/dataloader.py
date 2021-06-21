import os
import copy
import time
import random
import pickle

import pandas as pd
import numpy as np

import torch

from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# feature 추가 #1~2 수정

class Preprocess:
    def __init__(self, args):
        self.args = args

        self.args.cate_cols = []
        self.args.cont_cols = []
        self.args.features = []
        self.args.n_cols = {}

        self.train_data = None
        self.test_data = None


    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.8, shuffle=True, seed=42):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)
            random.shuffle(data)

        size = int(len(data) * ratio)

        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __feature_engineering(self, df):
        #1-1 categorical feature

        df['head'] = df.assessmentItemID.apply(lambda x: x[:4])
        df['mid'] = df.assessmentItemID.apply(lambda x: x[4:7])
        df['tail'] = df.assessmentItemID.apply(lambda x: x[7:])

        df['head_tail'] = df.assessmentItemID.apply(lambda x: x[:4]+x[7:])
        df['mid_tail'] = df.assessmentItemID.apply(lambda x: x[4:])

        #1-2 continuous feature

        ## time to sec
        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)

        ## find boundary
        # userID, testId 별 푼 문항의 누적 합
        df['UserTestCumtestnum'] = df.groupby(['userID','testId'])['answerCode'].cumcount()
        testId2maxlen = df[['assessmentItemID', 'testId']].drop_duplicates().groupby('testId').size()
        
        # test의 문항 수
        df['TestSize'] = df.testId.map(testId2maxlen)
        
        # user가 같은 test를 여러 번 푼 것인지 나타낸 변수 (처음 품 : 0, 두번 품 : 1, 세번 품 : 2)
        df['Retest'] = df['UserTestCumtestnum'] // df['TestSize']

        # boundary
        df['boundary'] = [u % t if t != 0 else 0.0 for t, u in zip(df['TestSize'], df['UserTestCumtestnum'])] 

        # 처음 푼 문제만 사용
        df = copy.deepcopy(df[df['Retest'] == 0])

        ## time diff
        time_diff = df.groupby(['userID', 'head', 'mid'])['Timestamp'].diff()
        df['time_diff'] = time_diff
        df.loc[df['boundary'] == 0, 'time_diff'] = np.NaN

        # 2번째 문제를 기준으로 1번째 문제를 채운다
        df['time_diff'].fillna(method='bfill', inplace=True)
        # df['time_diff'].fillna(0, inplace=True) -> 성능하락

        df['time_diff'] = df['time_diff'].map(lambda x: 600 if x>600 else x)
        df['time_diff'] = pd.cut(df['time_diff'], bins=600).astype(str) 
        

        # head별 정답률
        answer_head_mean = df.groupby(['userID', 'head'])['answerCode'].mean()
        answer_head_mean = answer_head_mean.reset_index(level=['userID', 'head'])
        answer_head_mean.columns = ['userID', 'head', 'head_answerProb']

        df = pd.merge(df, answer_head_mean, on=['userID', 'head'], how='left')

        # mid별 정답률
        answer_mid_mean = df.groupby(['userID', 'head', 'mid'])['answerCode'].mean()
        answer_mid_mean = answer_mid_mean.reset_index(level=['userID', 'head', 'mid'])
        answer_mid_mean.columns = ['userID', 'head', 'mid', 'mid_answerProb']

        df = pd.merge(df, answer_mid_mean, on=['userID', 'head', 'mid'], how='left')
        
        # tail별 정답률
        answer_tail_mean = df.groupby(['head', 'mid', 'tail'])['answerCode'].mean()
        answer_tail_mean = answer_tail_mean.reset_index(level=['head', 'mid', 'tail'])
        answer_tail_mean.columns = ['head', 'mid', 'tail', 'tail_answerProb']

        df = pd.merge(df, answer_tail_mean, on=['head', 'mid', 'tail'], how='left')

        #2 self.args.features의 순서와 trainer #1의 순서를 맞춰주자!
        # correct, question, test, tag, time_diff, head, mid, tail, mid_tail, 
        # head_answerProb, mid_answerProb, tail_answerProb, mask = batch

        self.args.cate_cols.extend([
            'assessmentItemID', 
            'testId', 
            'KnowledgeTag',
            'time_diff',
            'head',
            'mid',
            'tail',
            'mid_tail',
            ])

        self.args.cont_cols.extend([
            'head_answerProb',
            'mid_answerProb',
            'tail_answerProb',
            ])

        self.args.features.extend(
            ['answerCode'] + 
            self.args.cate_cols + 
            self.args.cont_cols
            )

        return df

    def __preprocessing(self, df, is_train = True):
        cate_cols = self.args.cate_cols

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test


        cont_cols = self.args.cont_cols

        # standard scaler
        std_scaler = preprocessing.StandardScaler().fit(df[cont_cols] )
        df[cont_cols] = std_scaler.transform(df[cont_cols])

        return df


    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
        
        cols = df.columns.tolist()
        for col in cols:
            if col in self.args.cont_cols:
                self.args.n_cols[col] = len(df[col].unique())
            
            if col in self.args.cate_cols:
                self.args.n_cols[col] = len(np.load(os.path.join(self.args.asset_dir, f'{col}_classes.npy')))

        df = df.sort_values(by=['userID','Timestamp'], axis=0)

        feature_columns = self.args.features        
        
        def get_values(cols, r):
            result = []
            for col in cols:
                result.append(r[col].values)

            return result

        if is_train:
            group = df.groupby(['userID', 'head', 'mid']).apply(
                lambda r: (get_values(feature_columns, r)))
            
        else:
            group = df.groupby('userID').apply(
                lambda r: (get_values(feature_columns, r)))

        if is_train:
            # save
            mass = (self.args.cate_cols, self.args.cont_cols, self.args.features)
            with open('/opt/ml/code/dkt/pkl/mass.pkl', 'wb') as f:
                pickle.dump(mass, f, pickle.HIGHEST_PROTOCOL)

            save_name = f'{self.args.pkl_dir}/{self.args.pkl_name}'
            with open(save_name, 'wb') as f:
                pickle.dump((group.values, self.args.n_cols), f, pickle.HIGHEST_PROTOCOL)

        return group.values

    def load_train_data(self, file_name):
        if self.args.pkl:
            # load
            pkl_name = f'{self.args.pkl_dir}/{self.args.pkl_name}'
            with open(pkl_name, 'rb') as f:
                self.train_data, self.args.n_cols = pickle.load(f)
            
            with open('/opt/ml/code/dkt/pkl/mass.pkl', 'rb') as f:
                self.args.cate_cols, self.args.cont_cols, self.args.features = pickle.load(f)
        else:
            self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # groupby 한 data라서 길이가 똑같기 때문에 row[0] 하고만 비교해도 된다
        seq_len = len(row[0])

        cols = list(row)

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cols):
                cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cols):
            cols[i] = torch.tensor(col)

        return cols

    def __len__(self):
        return len(self.data)


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = True
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader