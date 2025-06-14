import numpy as np
import math
import scipy.stats
import scipy.special
import pandas as pd
import os
import json
from sklearn.datasets import make_moons

class ToyData:
    def __init__(self, dataset_name: str, dataset_dir_name: str):
        self.dataset_name = dataset_name
        
        self.dataset_dir_name = dataset_dir_name
                
        self.create_save_path()

    def create_save_path(self):
        abs_path = os.path.abspath(os.getcwd())
        abs_path_dir = os.path.join(abs_path, self.dataset_dir_name, self.dataset_name)
        
        if not os.path.exists(abs_path_dir):
            os.makedirs(abs_path_dir)
        else:
            print(f"Directory {abs_path_dir} already exists")
        
        self.abs_path_dir = abs_path_dir
        
    @staticmethod
    def create_dataset(**kwargs) -> pd.DataFrame:
        raise NotImplementedError("create_dataset method must be implemented")
    
    def save_dataset(
        self,
        dataset_kwargs: dict,
        ):
        raise NotImplementedError("save_dataset method must be implemented")
                
class ToyClassificationData(ToyData):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name, dataset_dir_name="datasets_toy_classification")
        
    def save_dataset(
        self,
        dataset_kwargs: dict,
        ):
            
            dataset = self.create_dataset(**dataset_kwargs)
            
            dataset.to_csv(os.path.join(self.abs_path_dir, "data.csv"))
            
            # unique labels
            labels = dataset["label"].unique()
            
            # string to int mapping
            label_map = {str(label): int(label) for label in labels}
            
            print("Labels distribution")
            print(dataset["label"].value_counts())
            with open(os.path.join(self.abs_path_dir, "info.json"), "w") as f:
                json.dump(
                    {
                        **dataset_kwargs,
                        "feature_columns": [column for column in list(dataset.columns) if column != "label"],
                        "label": "y",
                        "map": label_map,
                    }, f)
                
class ToyRegressionData(ToyData):
    def __init__(self, dataset_name: str):        
        super().__init__(dataset_name, dataset_dir_name = "datasets_toy_regression")
    
    def save_dataset(
        self,
        dataset_kwargs: dict,
        ):
            
            dataset = self.create_dataset(**dataset_kwargs)
            
            dataset.to_csv(os.path.join(self.abs_path_dir, "data.csv"))
            
            with open(os.path.join(self.abs_path_dir, "info.json"), "w") as f:
                json.dump(
                    {
                        **dataset_kwargs,
                        "feature_columns": [column for column in list(dataset.columns) if column != "label"],
                        "label": "y",
                        "map": {},
                    }, f)
                
# Classification Datasets

class LogisticRegressionData(ToyClassificationData):
    @staticmethod
    def create_normal_features(num_features: int, feature_dim: int, feature_means: np.ndarray, feature_stds: np.ndarray, round_dp: int=1, seed:int=0):
        if len(feature_means) != feature_dim:
            raise ValueError("feature_means must have length equal to feature_dim")
        if len(feature_stds) != feature_dim:
            raise ValueError("feature_stds must have length equal to feature_dim")
        
        x = scipy.stats.norm.rvs(size=(num_features, feature_dim), loc=feature_means, scale=feature_stds, random_state=seed)
        
        return np.round(x, round_dp)
    
    @staticmethod
    def create_labels(features: np.ndarray, bias: float, coefficients: np.ndarray, seed:int=0):
        logits = np.matmul(features, coefficients) + bias
        
        probabilities = scipy.special.expit(logits)
            
        y = scipy.stats.bernoulli.rvs(probabilities, size=len(probabilities), random_state=seed)
        
        return y

    @staticmethod
    def create_pandas_dataset(features: np.ndarray, y: np.ndarray):
        data_dict = {f"x{i+1}": features[:,i] for i in range(features.shape[1])}
        data_dict.update({"label": y})
        
        dataset = pd.DataFrame(data_dict).rename_axis("index")
        
        return dataset
    
    @staticmethod
    def create_dataset(
        feature_dimensions: int = 1,
        feature_means: list[float] = [0.0],
        feature_stds: list[float] = [1.0],
        bias: float = 0.0,
        coefficients: list[float] = [0.0],
        dataset_size = 100,
        seed: int = 0,
        round_dp: int = 1
        ):
            
            x = LogisticRegressionData.create_normal_features(dataset_size, feature_dimensions, np.array(feature_means), np.array(feature_stds), seed=seed, round_dp=round_dp)
                
            y = LogisticRegressionData.create_labels(x, bias, np.array(coefficients), seed=seed)
            
            dataset = LogisticRegressionData.create_pandas_dataset(x, y)
            
            print(dataset.head(20))
            
            print("Labels distribution")
            print(dataset["label"].value_counts())
            
            return dataset
        
class MoonsData(ToyClassificationData):
    @staticmethod
    def create_dataset(
        dataset_size: int = 100,
        noise: float = 0.1,
        round_dp: int = 2,
        seed: int = 0
        ):
            
            x, y = make_moons(n_samples=dataset_size, noise=noise, random_state=seed)
            
            dataset = pd.DataFrame({"x1": np.round(x[:,0], round_dp), "x2": np.round(x[:,1], round_dp), "label": y}).rename_axis("index")
            
            print(dataset.head(20))
            
            print("Labels distribution")
            print(dataset["label"].value_counts())
            
            return dataset
        
class SpiralData(ToyClassificationData):
    '''Code adapted from https://github.com/corneauf/N-Arm-Spiral-Dataset'''
    @staticmethod
    def rotate_point(point, angle):
        """Rotate two point by an angle.

        Parameters
        ----------
        point: 2d numpy array
            The coordinate to rotate.
        angle: float
            The angle of rotation of the point, in degrees.

        Returns
        -------
        2d numpy array
            Rotated point.
        """
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_point = rotation_matrix.dot(point)
        return rotated_point
    
    def generate_spiral(self, samples, start, end, angle, noise):
        """Generate a spiral of points.

        Given a starting end, an end angle and a noise factor, generate a spiral of points along
        an arc.

        Parameters
        ----------
        samples: int
            Number of points to generate.
        start: float
            The starting angle of the spiral in degrees.
        end: float
            The end angle at which to rotate the points, in degrees.
        angle: float
            Angle of rotation in degrees.
        noise: float
            The noisyness of the points inside the spirals. Needs to be less than 1.
        """
        # Generate points from the square root of random data inside an uniform distribution on [0, 1).
        points = math.radians(start) + np.sqrt(self.numpy_rng.random((samples, 1))) * math.radians(end)

        # Apply a rotation to the points.
        rotated_x_axis = np.cos(points) * points + self.numpy_rng.random((samples, 1)) * noise
        rotated_y_axis = np.sin(points) * points + self.numpy_rng.random((samples, 1)) * noise

        # Stack the vectors inside a samples x 2 matrix.
        rotated_points = np.column_stack((rotated_x_axis, rotated_y_axis))
        return np.apply_along_axis(self.rotate_point, 1, rotated_points, math.radians(angle))

    def create_dataset(self, dataset_size: int, arms: int = 3, start: int = 0, end: int = 720, noise: float = 1.2, seed: int = 0, round_dp: int = 2):
        # Create a list of the angles at which to rotate the arms.
        # Either we find the angles automatically by dividing by the number of arms
        # Or we just use the angle given by the user.
        SCALE = 4
        
        self.numpy_rng = np.random.default_rng(seed)
        
        classes = np.empty((0, 3))
        angles = [(360 / arms)* i for i in range(arms)]

        for i, angle in enumerate(angles):
            points = np.round(self.generate_spiral(dataset_size // arms, start, end, angle, noise)/SCALE, round_dp)
            classified_points = np.hstack((points, np.full((dataset_size // arms, 1), i)))
            classes = np.concatenate((classes, classified_points))
        
        dataset = pd.DataFrame(classes, columns=["x1", "x2", "label"]).rename_axis("index")  
        dataset["label"] = dataset["label"].astype(int)
        
        return dataset
          
# Regression Datasets
        
class LinearRegressionData(ToyRegressionData):
    @staticmethod
    def create_normal_features(num_features: int, feature_dim: int, feature_means: np.ndarray, feature_stds: np.ndarray, round_dp: int=1, seed:int=0):
        if len(feature_means) != feature_dim:
            raise ValueError("feature_means must have length equal to feature_dim")
        if len(feature_stds) != feature_dim:
            raise ValueError("feature_stds must have length equal to feature_dim")
        
        x = scipy.stats.norm.rvs(size=(num_features, feature_dim), loc=feature_means, scale=feature_stds, random_state=seed)
        
        return np.round(x, round_dp)
    
    @staticmethod
    def create_labels(features: np.ndarray, bias: float, coefficients: np.ndarray, noise_std: float, round_dp: int=1, seed:int=0):
        y = np.matmul(features, coefficients) + bias
        
        noise = scipy.stats.norm.rvs(size=len(y), loc=0, scale=noise_std, random_state=seed+1)
        y += noise
        
        return np.round(y, round_dp)

    @staticmethod
    def create_pandas_dataset(features: np.ndarray, y: np.ndarray):
        data_dict = {f"x{i+1}": features[:,i] for i in range(features.shape[1])}
        data_dict.update({"label": y})
        
        dataset = pd.DataFrame(data_dict).rename_axis("index")
        
        return dataset
    
    @staticmethod
    def create_dataset(
        feature_dimensions: int = 1,
        feature_means: list[float] = [0.0],
        feature_stds: list[float] = [1.0],
        bias: float = 0.0,
        coefficients: list[float] = [0.0],
        noise_std: float = 0.1,
        dataset_size = 100,
        seed: int = 0,
        round_dp: int = 1
        ):
            
            x = LinearRegressionData.create_normal_features(dataset_size, feature_dimensions, np.array(feature_means), np.array(feature_stds), seed=seed, round_dp=round_dp)
                
            y = LinearRegressionData.create_labels(x, bias, np.array(coefficients), noise_std, round_dp=round_dp)
            
            dataset = LinearRegressionData.create_pandas_dataset(x, y)
            
            print(dataset.head(20))
            
            return dataset

class VaryingLinearNoise(ToyRegressionData):
    '''Gaps Dataset'''
    @staticmethod
    def create_features(num_features_per_mode: np.ndarray, mode_means: np.ndarray, mode_stds: np.ndarray, round_dp: int=1, seed:int=0):
        if len(mode_means) != len(num_features_per_mode):
            raise ValueError("mode_means must have length equal to num_features_per_mode")
        if len(mode_stds) != len(num_features_per_mode):
            raise ValueError("mode_stds must have length equal to num_features_per_mode")
        
        x = []
        for i in range(len(num_features_per_mode)):
            x.append(
                np.round(scipy.stats.norm.rvs(
                    size=(num_features_per_mode[i], 1),
                    loc=mode_means[i],
                    scale=mode_stds[i],
                    random_state=seed+i
                    ), round_dp).squeeze()
            )
                
        return x
    
    @staticmethod
    def create_labels(features: list[np.ndarray], mode_biases: list[float], mode_coeffs: list[float], noise_stds: list[float], round_dp: int=1, seed:int=0):
        y = []
        for i in range(len(features)):
            y_value = features[i] * mode_coeffs[i] + mode_biases[i]

            y_value += scipy.stats.norm.rvs(
                size=len(y_value),
                loc=0,
                scale=noise_stds[i],
                random_state=len(features) + seed + i
            )
            
            y.append(np.round(y_value, round_dp).squeeze())
    
        return y
    
    @staticmethod
    def create_pandas_dataset(features: np.ndarray, y: np.ndarray):
        data_dict = {"x": features, "label": y}
        
        dataset = pd.DataFrame(data_dict).rename_axis("index")
        
        return dataset
    
    @staticmethod
    def create_dataset(
        num_features_per_mode: list[int] = [50, 50],
        mode_means: list[float] = [0.0, 3.0],
        mode_stds: list[float] = [1.0, 1.0],
        mode_biases: list[float] = [0.0, 0.0],
        mode_coeffs: list[float] = [1.0, 1.0],
        noise_stds: list[float] = [0.1, 0.1],
        seed: int = 1,
        round_dp: int = 1
        ):
            
            x = VaryingLinearNoise.create_features(np.array(num_features_per_mode), np.array(mode_means), np.array(mode_stds), seed=seed, round_dp=round_dp)
                
            print(noise_stds)
            
            y = VaryingLinearNoise.create_labels(x, mode_biases, mode_coeffs, noise_stds, round_dp=round_dp, seed=seed)

            x_stacked = np.concatenate(x).squeeze()
            y_stacked = np.concatenate(y).squeeze()
            
            dataset = VaryingLinearNoise.create_pandas_dataset(x_stacked, y_stacked)
            
            print(dataset.head(20))
            
            return dataset