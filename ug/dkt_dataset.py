import logging
import os.path as p
from typing import Callable

import torch
import easydict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from logger import get_logger

from torch.utils.data import Sampler


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count: dict = {}

        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] = label_to_count.get(label, 0) + 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class Preprocess:
    def __init__(self, args, fe_pipeline, columns):
        self.args = args
        self.datas = {"train": None, "valid": None, "test": None}
        self.fe_pipeline = fe_pipeline
        self.columns = columns

        self.logger = get_logger("preprocess")
        self.logger.setLevel(logging.INFO)

        assert args.data_dir, f"{args.data_dir}??? ??????????????????."
        self.datas["train"] = pd.read_csv(p.join(args.data_dir, "train_data.csv"))
        self.datas["test"] = pd.read_csv(p.join(args.data_dir, "test_data.csv"))

        self.datas["org_train"] = self.datas["train"].copy()
        self.datas["org_test"] = self.datas["test"].copy()

        log_file_handler = logging.FileHandler(p.join(self.args.root_dir, "preprocess.log"))
        self.logger.addHandler(log_file_handler)
        self.logger.addHandler(logging.StreamHandler())

    def get_data(self, key="train"):
        """ return datas """
        assert key in self.datas.keys(), f"{key} not in {self.datas.keys()}"
        assert self.datas[key] is not None, f"{key} dataset is None"
        return self.datas[key]

    def split_data(self, seed=42, test_size=0.3):
        """ User ???????????? Split ??????. """
        self.logger.info("Split based on User")
        self.logger.info(f"Original Train Dataset: {len(self.datas['train'])}")

        user_id = self.datas["train"].userID.unique()
        t_idx, v_idx = train_test_split(user_id, test_size=test_size, random_state=seed)
        t_idx = set(t_idx)

        fancy_index = self.datas["train"]["userID"].apply(lambda x: x in t_idx)
        train = self.datas["train"][fancy_index]
        valid = self.datas["train"][(fancy_index - 1).astype(bool)]

        self.datas["train"] = train
        self.datas["valid"] = valid

        self.logger.info(f"Split Train Dataset: {len(self.datas['train'])}")
        self.logger.info(f"Split Valid Dataset: {len(self.datas['valid'])}")

    def _make_rows_to_sequence(self, r):
        datas = []
        for key in self.columns:
            if key == "userID":
                continue
            datas.append(r[key].values)
        return tuple(datas)

    def _data_augmentation_test_dataset(self):
        """ 1 ????????? ???????????? ?????? """

        self.logger.info("Use the test datast for data augmentation")
        self.logger.info(f"Before Length: {len(self.datas['train'])}")

        test_dataset = self.datas["test"]
        aug_test_dataset = test_dataset[test_dataset["answerCode"] != -1]
        self.datas["train"] = pd.concat([self.datas["train"], aug_test_dataset], axis=0)

        self.logger.info(f"After Length: {len(self.datas['train'])}")

    def _data_augmentation_user_testpaper(self):
        """ 2 ????????? ?????? """

        self.logger.info("Group By (userID, testPaper)")
        t_grouped = self.datas["train"].groupby(["userID", "testPaper"]).apply(lambda r: self._make_rows_to_sequence(r))
        v_grouped = self.datas["valid"].groupby(["userID", "testPaper"]).apply(lambda r: self._make_rows_to_sequence(r))
        tt_grouped = self.datas["test"].groupby(["userID", "testPaper"]).apply(lambda r: self._make_rows_to_sequence(r))

        self.datas["train_grouped"] = t_grouped.values
        self.datas["valid_grouped"] = v_grouped.values
        self.datas["test_grouped"] = tt_grouped.values

        self.logger.info(f"Group By (userID, testPaper) Train Length: {len(self.datas['train_grouped'])}")
        self.logger.info(f"Group By (userID, testPaper) Valid Length: {len(self.datas['valid_grouped'])}")
        self.logger.info(f"Group By (userID, testPaper) Test Length: {len(self.datas['test_grouped'])}")

    def _data_augmentation_user_testid(self):
        """ 3 ????????? ?????? """

        self.logger.info("Group By (userID, firstClass)")
        t_grouped = (
            self.datas["train"].groupby(["userID", "firstClass"]).apply(lambda r: self._make_rows_to_sequence(r))
        )
        v_grouped = (
            self.datas["valid"].groupby(["userID", "firstClass"]).apply(lambda r: self._make_rows_to_sequence(r))
        )
        tt_grouped = (
            self.datas["test"].groupby(["userID", "firstClass"]).apply(lambda r: self._make_rows_to_sequence(r))
        )

        self.datas["train_grouped"] = t_grouped.values
        self.datas["valid_grouped"] = v_grouped.values
        self.datas["test_grouped"] = tt_grouped.values

        self.logger.info(f"Group By (userID, firstClass) Length: {len(self.datas['train_grouped'])}")
        self.logger.info(f"Group By (userID, firstClass) Length: {len(self.datas['valid_grouped'])}")
        self.logger.info(f"Group By (userID, firstClass) Length: {len(self.datas['test_grouped'])}")

    def data_augmentation(self, choices=[1, 2]):
        """???????????? ???????????? ????????? ????????? ?????????.
        1. ????????? ????????? ??? ???????????? ???
        2. UserID + testPaper
        3. UserID + firstClass
        """

        if 2 in choices and 3 in choices:
            raise ValueError("2, 3??? ?????? ?????? ????????????..!")

        data_aug = {
            1: self._data_augmentation_test_dataset,
            2: self._data_augmentation_user_testpaper,
            3: self._data_augmentation_user_testid,
        }

        for choice in choices:
            data_aug[choice]()

    def feature_engineering(self):
        """ Feature engineering??? ???????????????. """
        for key in ["train", "test"]:
            assert self.datas[key] is not None, f"you must loads {key} dataset"
            #  is_train = True if key == "train" else False
            df = self.fe_pipeline.transform(self.datas[key], key=key)
            self.datas[key] = df[self.columns]

    def scaling(self, pre_encoders):
        """????????? ?????? ??????????????? ??????, "train", "test"??? ???????????? ??????
        label   : LabelEncoder
        min_max : MinMaxScaler
        std     : StandardScaler
        """
        assert (
            len(set(pre_encoders.keys()).intersection(set(["label", "min_max", "std"]))) == 3
        ), "pre_encoders??? key?????? ('label', 'min_max', 'std') ??? ??????????????????."

        encoders = {}
        self.args.n_embeddings = easydict.EasyDict()
        self.args.n_linears = list()

        self.logger.info("Preprocessing Labels .. ")
        self.logger.info(f"Label Columns: {pre_encoders['label']}")

        for k in pre_encoders["label"]:
            encoders[k] = LabelEncoder()
            labels = self.datas["train"][k].unique().tolist() + ["unknown"]
            self.args.n_embeddings[k] = len(labels)

            # TODO: Label HistoGram ?????????..?
            self.logger.info(f"\nLength of {k:<20} : {len(labels)}")

            # Train Fit Transform
            encoders[k].fit(labels)
            self.datas["train"][k] = self.datas["train"][k].astype(str)
            self.logger.info(f"Before : {self.datas['train'][k][:10]}")

            self.datas["train"][k] = encoders[k].transform(self.datas["train"][k])
            self.logger.info(f"After : {self.datas['train'][k][:10]}")

            # Valid Transform
            self.datas["valid"][k] = self.datas["valid"][k].apply(lambda x: x if x in labels else "unknown")
            self.datas["valid"][k] = self.datas["valid"][k].astype(str)
            self.datas["valid"][k] = encoders[k].transform(self.datas["valid"][k])

            # Test Transform
            self.datas["test"][k] = self.datas["test"][k].apply(lambda x: x if x in labels else "unknown")
            self.datas["test"][k] = self.datas["test"][k].astype(str)
            self.datas["test"][k] = encoders[k].transform(self.datas["test"][k])

        self.logger.info("Preprocessing Min Max .. ")
        self.logger.info(f"Min Max Columns: {pre_encoders['min_max']}")

        if pre_encoders["min_max"]:
            mm_cols = pre_encoders["min_max"]
            mm_encoder = MinMaxScaler()
            self.args.n_linears += mm_cols

            # Train Fit Transform
            self.datas["train"][mm_cols] = mm_encoder.fit_transform(self.datas["train"][mm_cols])
            self.logger.info(f"MAX: {mm_encoder.data_max_} MIN: {mm_encoder.data_min_}")

            # Valid Transform
            self.datas["valid"][mm_cols] = mm_encoder.transform(self.datas["valid"][mm_cols])

            # Test Transform
            self.datas["test"][mm_cols] = mm_encoder.transform(self.datas["test"][mm_cols])

        self.logger.info("Preprocessing Min Max .. ")
        self.logger.info(f"Standard Columns: {pre_encoders['std']}")

        if pre_encoders["std"]:
            std_cols = pre_encoders["std"]
            std_encoder = StandardScaler()
            self.args.n_linears += std_cols

            # Train Fit Transform
            self.datas["train"][std_cols] = std_encoder.fit_transform(self.datas["train"][std_cols])
            self.logger.info(f"MEAN: {std_encoder.mean_} VAR: {std_encoder.var_}")

            # Valid Transform
            self.datas["valid"][std_cols] = std_encoder.transform(self.datas["valid"][std_cols])

            # Test Transform
            self.datas["test"][std_cols] = std_encoder.transform(self.datas["test"][std_cols])

    def preprocessing(self, pre_encoders):
        self._data_augmentation_test_dataset()
        self.feature_engineering()
        self.scaling(pre_encoders)
        self.split_data()
        self._data_augmentation_user_testid()


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, columns, is_train=True):
        self.data = data
        self.args = args
        self.columns = columns
        self.is_train = is_train

    def __getitem__(self, index):
        row = self.data[index]

        # ??? data??? sequence length
        seq_len = len(row[0])
        datas = {key: row[idx] for idx, key in enumerate(self.columns)}

        # max seq len??? ??????????????? ????????? ?????? ????????? ?????? ?????? Random??? Sequence??? ????????????.
        if seq_len > self.args.max_seq_len:
            if self.args.use_dynamic is True and self.is_train is True:
                s_idx = np.random.randint(0, seq_len - self.args.max_seq_len + 1)
                for key, value in datas.items():
                    datas[key] = value[s_idx : s_idx + self.args.max_seq_len]
            else:
                for key, value in datas.items():
                    datas[key] = value[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask??? columns ????????? ????????????
        datas["mask"] = mask

        # np.array -> torch.tensor ?????????
        for key, value in datas.items():
            datas[key] = torch.tensor(value)

        return datas

    def __len__(self):
        return len(self.data)
