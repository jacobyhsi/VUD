from datasets import load_from_disk
from datasets import load_dataset as load
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
from src.utils import ToyDataUtils

def load_dataset(
    data_path,
    data_type='tabular',
    data_split_seed=123
    ) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    # Load Dataset
    dataset: Dataset = DATATYPE_TO_DATACLASS[data_type](
        data_path=data_path,
        data_split_seed=data_split_seed,
        )
    data = dataset.get_train_data()
    test = dataset.get_test_data()
    # exit() # inspect feature names

    # Load Dataset Configs
    file_config = json.load(open(f'{data_path}/info.json'))
    label_map = file_config['map']
    label_keys = list(label_map)

    return data, test, label_keys

class Dataset():
    def __init__(self, data_path, data_split_seed: int = 123):
        data = self.load_data(data_path)
        # Split data
        self.data, self.test_data = train_test_split(data, test_size=0.2, random_state=data_split_seed)

    def get_train_data(self):
        return self.data

    def get_test_data(self):
        return self.test_data
    
    def load_data(self, data_path):
        raise NotImplementedError
class ToyClassificationDataset(Dataset):
    def load_data(self, data_path: str):
        data = pd.read_csv(os.path.join(data_path, 'data.csv'), index_col=0)
                    
        data['label'] = data['label'].astype(int)
        
        feature_column = ToyDataUtils.get_feature_columns(data)
        
        data['note'] = data.apply(
            lambda row: ToyDataUtils.parse_features_to_note(row, feature_column),
            axis=1
        )
        
        return data
    
class ToyRegressionDataset(Dataset):
    def load_data(self, data_path: str):
        data = pd.read_csv(os.path.join(data_path, 'data.csv'), index_col=0)
                    
        data['label'] = data['label'].astype(float)
        
        feature_column = ToyDataUtils.get_feature_columns(data)
        
        data['note'] = data.apply(
            lambda row: ToyDataUtils.parse_features_to_note(row, feature_column),
            axis=1
        )
        
        return data  
     
class QADataset:
    def __init__(self, id_dataset, ood_dataset, dataset_size:int = 150, test_size:float = 0.8, seed: int = 123):
        self.id_dataset = id_dataset
        self.ood_dataset = ood_dataset
        self.dataset_size = dataset_size
        self.test_size = test_size
        self.seed = seed

    def load_data(self):
        # Load ID dataset
        if self.id_dataset == "boolqa":
            train_id, test_id = self._load_boolqa()
        elif self.id_dataset == "hotpotqa":
            train_id, test_id = self._load_hotpotqa()
        elif self.id_dataset == "pubmedqa":
            train_id, test_id = self._load_pubmedqa()
        else:
            raise ValueError(f"Unknown ID dataset: {self.id_dataset}")

        # Load OOD dataset (only test portion needed)
        if self.ood_dataset == "boolqa":
            _, test_ood = self._load_boolqa()
        elif self.ood_dataset == "hotpotqa":
            _, test_ood = self._load_hotpotqa()
        elif self.ood_dataset == "pubmedqa":
            _, test_ood = self._load_pubmedqa()
        else:
            raise ValueError(f"Unknown OOD dataset: {self.ood_dataset}")

        # Convert label keys to strings from ID training set
        label_keys = [str(k) for k in sorted(train_id['label'].unique().tolist())]

        return train_id.reset_index(drop=True), test_id.reset_index(drop=True), test_ood.reset_index(drop=True), label_keys

    def _load_boolqa(self):
        ds = load("boolq")                          
        df_train = pd.DataFrame(ds["train"])
        df_val = pd.DataFrame(ds["validation"])
        df_all = pd.concat([df_train, df_val], ignore_index=True)

        df_all = df_all.sample(n=self.dataset_size, random_state=self.seed).reset_index(drop=True)

        df_all = df_all.rename(columns={"answer": "label"})
        df_all["label"] = df_all["label"].astype(int)

        df_all["note"] = (
            "Question: " + df_all["question"].str.lower() + " Context: " + df_all["passage"].str.lower()
        )

        df_all = df_all[["note", "label"]]

        train_df, test_df = train_test_split(
            df_all,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df_all["label"],
        )

        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )

    def _load_hotpotqa(self):
        ds = load("hotpot_qa", "fullwiki", trust_remote_code=True)
        df_all = pd.concat(
            [ds[split].to_pandas() for split in ds.keys()],
            ignore_index=True
        )

        df = df_all[df_all["answer"].isin(["yes", "no"])].copy()
        df = df.sample(n=self.dataset_size, random_state=self.seed).reset_index(drop=True)

        df["label"] = df["answer"].map({"no": 0, "yes": 1}).astype(int)

        def _get_support_titles(sf):
            if isinstance(sf, dict) and "title" in sf:
                return list(sf["title"])
            return [item[0] for item in sf]

        df["support_titles"] = df["supporting_facts"].apply(_get_support_titles)

        def _flatten_supporting_context(ctx, support_titles):
            """
            ctx is a dict: {'title': np.ndarray, 'sentences': np.ndarray of np.ndarrays}
            Return lowercase string of sentences whose title is in support_titles.
            """
            try:
                titles = ctx["title"]
                sentences_blocks = ctx["sentences"]
                return " ".join(
                    sent
                    for title, block in zip(titles, sentences_blocks)
                    if title in support_titles
                    for sent in block
                ).lower()
            except Exception as e:
                print("Context parsing error:", ctx)
                raise e

        df["context_str"] = df.apply(
            lambda row: _flatten_supporting_context(row["context"], row["support_titles"]),
            axis=1
        )

        df["note"] = "Question: " + df["question"].str.lower() + " Context: " + df["context_str"]
        df_final = df[["note", "label"]]

        train_df, test_df = train_test_split(
            df_final,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df_final["label"],
        )

        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )

    def _load_pubmedqa(self):
        pqa = load("pubmed_qa", "pqa_labeled")
        pqa_df = pd.DataFrame(pqa["train"])

        pqa_df = pqa_df[pqa_df["final_decision"].isin(["yes", "no"])].copy()

        def take_passage(row):
            la = row.get("long_answer", "")
            if isinstance(la, str) and la.strip():
                return la.strip()
            ctx = row.get("context", [])
            return ctx[0].strip() if isinstance(ctx, list) and ctx else ""

        pqa_df["passage"] = pqa_df.apply(take_passage, axis=1)

        pqa_df = pqa_df.sample(n=self.dataset_size, random_state=self.seed).reset_index(drop=True)

        pqa_df["question"] = pqa_df["question"].str.strip()
        pqa_df["note"] = (
            "Question: " + pqa_df["question"].str.lower() +
            " Context: " + pqa_df["passage"].str.lower()
        )
        pqa_df["label"] = pqa_df["final_decision"].map({"no": 0, "yes": 1}).astype(int)

        df_final = pqa_df[["note", "label"]]

        train_df, test_df = train_test_split(
            df_final,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df_final["label"]
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        
DATATYPE_TO_DATACLASS: dict[str, Dataset] = {
    "toy_classification": ToyClassificationDataset,
    "toy_regression": ToyRegressionDataset,
}